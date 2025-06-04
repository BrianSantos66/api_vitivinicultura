from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import requests
import jwt
from datetime import datetime, timedelta, timezone
import uvicorn
from functools import lru_cache
import logging
import re

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações JWT
SECRET_KEY = "123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# URLs dos dados da Embrapa
EMBRAPA_URLS = {
    "producao": "http://vitibrasil.cnpuv.embrapa.br/download/Producao.csv",
    "processamento_viniferas": "http://vitibrasil.cnpuv.embrapa.br/download/ProcessaViniferas.csv",
    "processamento_americanas": "http://vitibrasil.cnpuv.embrapa.br/download/ProcessaAmericanas.csv",
    "processamento_mesa": "http://vitibrasil.cnpuv.embrapa.br/download/ProcessaMesa.csv",
    "processamento_semclass": "http://vitibrasil.cnpuv.embrapa.br/download/ProcessaSemclass.csv",
    "comercializacao": "http://vitibrasil.cnpuv.embrapa.br/download/Comercio.csv",
    "importacao_vinhos": "http://vitibrasil.cnpuv.embrapa.br/download/ImpVinhos.csv",
    "importacao_espumantes": "http://vitibrasil.cnpuv.embrapa.br/download/ImpEspumantes.csv",
    "importacao_frescas": "http://vitibrasil.cnpuv.embrapa.br/download/ImpFrescas.csv",
    "importacao_passas": "http://vitibrasil.cnpuv.embrapa.br/download/ImpPassas.csv",
    "importacao_suco": "http://vitibrasil.cnpuv.embrapa.br/download/ImpSuco.csv",
    "exportacao_vinhos": "http://vitibrasil.cnpuv.embrapa.br/download/ExpVinho.csv",
    "exportacao_espumantes": "http://vitibrasil.cnpuv.embrapa.br/download/ExpEspumantes.csv",
    "exportacao_uva": "http://vitibrasil.cnpuv.embrapa.br/download/ExpUva.csv",
    "exportacao_sucos": "http://vitibrasil.cnpuv.embrapa.br/download/ExpSuco.csv"
}

# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None

class UserInDB(User):
    hashed_password: str

class DataResponse(BaseModel):
    categoria: str
    total_registros: int
    dados: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class FilterParams(BaseModel):
    produto: Optional[str] = None
    pais: Optional[str] = None
    ano_inicio: Optional[int] = None
    ano_fim: Optional[int] = None
    limit: Optional[int] = 100

# =============================================================================
# MÓDULO DE AUTENTICAÇÃO
# =============================================================================

class AuthManager:
    """Gerenciador de autenticação"""
    
    def __init__(self):
        self.fake_users_db = {
            "admin": {
                "username": "admin",
                "email": "admin@embrapa-api.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # secret
            }
        }
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verificar senha (simplificado - use bcrypt em produção)"""
        return plain_password == "secret"
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Obter usuário do banco"""
        if username in self.fake_users_db:
            user_dict = self.fake_users_db[username]
            return UserInDB(**user_dict)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Autenticar usuário"""
        user = self.get_user(username)
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Criar token JWT"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

# =============================================================================
# MÓDULO DE PROCESSAMENTO DE DADOS
# =============================================================================

class EmbrapaDataProcessor:
    """Processador de dados da Embrapa"""
    
    @staticmethod
    def is_import_export_category(categoria: str) -> bool:
        """Verificar se a categoria é de importação ou exportação"""
        return categoria.startswith('importacao_') or categoria.startswith('exportacao_')
    
    @staticmethod
    def is_production_processing_category(categoria: str) -> bool:
        """Verificar se a categoria é de produção ou processamento"""
        return categoria.startswith('producao') or categoria.startswith('processamento_') or categoria == 'comercializacao'
    
    @staticmethod
    def clean_import_export_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpar dados de importação/exportação - versão ultra simplificada"""
        if df.empty:
            return df
        
        logger.info(f"Shape original: {df.shape}")
        logger.info(f"Colunas: {list(df.columns)}")
        
        logger.info("Primeiras 2 linhas do CSV:")
        for i in range(min(2, len(df))):
            logger.info(f"Linha {i}: {df.iloc[i].tolist()}")
        
        # Identificar posições das colunas
        year_positions = []  # Lista de posições das colunas de ano
        base_columns = []    # Lista de (posição, nome) das colunas base
        
        year_pattern = re.compile(r'^(19|20)\d{2}')
        
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            if year_pattern.match(col_str):
                year_positions.append(i)
                
            elif col_str.lower() not in []:
                base_columns.append((i, col_str))
        
        logger.info(f"Posições das colunas de anos: {year_positions}")
        logger.info(f"Colunas base: {[name for _, name in base_columns]}")
        
        if not year_positions:
            logger.error("Nenhuma coluna de ano encontrada!")
            return df
        
        result_data = []
        
        for row_idx, row in df.iterrows():
            row_data = {}

            for pos, name in base_columns:
                value = row.iloc[pos]
                column_name = name.lower() if name.lower() in ['id', 'control'] else name
                row_data[column_name] = value if pd.notna(value) else ''
            
            for i in range(0, len(year_positions), 2):
                pos_qtd = year_positions[i]
                ano = str(df.columns[pos_qtd]).strip()
                
                qtd = row.iloc[pos_qtd]
                qtd = float(qtd) if pd.notna(qtd) and str(qtd).strip() != '' else 0
                
                valor = 0
                if i + 1 < len(year_positions):
                    pos_val = year_positions[i + 1]
                    valor = row.iloc[pos_val]
                    valor = float(valor) if pd.notna(valor) and str(valor).strip() != '' else 0
                
                row_data[f"{ano}_quantidade_kg"] = qtd
                row_data[f"{ano}_valor_usd"] = valor

            
            result_data.append(row_data)
        
        result_df = pd.DataFrame(result_data)
        
        for col in result_df.columns:
            if '_quantidade_kg' in col or '_valor_usd' in col:
                try:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
                except:
                    result_df[col] = 0
        
        logger.info(f"Shape final: {result_df.shape}")
        logger.info(f"Colunas resultado: {list(result_df.columns)}")
        
        return result_df
    
    @staticmethod
    def clean_production_processing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpar dados de produção/processamento"""
        if df.empty:
            return df
        
        logger.info(f"Processando dados de produção/processamento. Shape original: {df.shape}")
        
        if 'cultivar' in df.columns:
            df = df.rename(columns={'cultivar': 'produto'})
        
        df.columns = [col.lower() if col.lower() in ['id', 'control'] else col for col in df.columns]
        
        logger.info(f"Dados limpos de produção/processamento. Shape final: {df.shape}")
        
        return df
    
    @classmethod
    def fetch_and_process_data(cls, categoria: str) -> pd.DataFrame:
        """Buscar e processar dados da Embrapa"""
        if categoria not in EMBRAPA_URLS:
            raise HTTPException(status_code=404, detail=f"Categoria {categoria} não encontrada")
        
        try:
            logger.info(f"Buscando dados da categoria: {categoria}")
            url = EMBRAPA_URLS[categoria]
            
            # Fazer requisição HTTP
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = response.text
            first_lines = '\n'.join(content.split('\n')[:5])
            
            if '\t' in first_lines and first_lines.count('\t') > first_lines.count(';'):
                delimiter = '\t'
            else:
                delimiter = ';'
            
            logger.info(f"Delimitador detectado: {'TAB' if delimiter == '\t' else 'SEMICOLON'}")
            
            try:
                df = pd.read_csv(url, delimiter=delimiter, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(url, delimiter=delimiter, encoding='latin-1')
            
            if df.empty:
                logger.warning(f"DataFrame vazio para categoria: {categoria}")
                return df
            
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                logger.warning(f"DataFrame vazio após limpeza inicial para categoria: {categoria}")
                return df
            
            # Processamento específico por tipo de categoria
            if cls.is_import_export_category(categoria):
                df = cls.clean_import_export_data(df)
            elif cls.is_production_processing_category(categoria):
                df = cls.clean_production_processing_data(df)
            
            logger.info(f"Dados processados com sucesso. Shape final: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados da categoria {categoria}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Erro ao buscar dados da Embrapa: {str(e)}"
            )

# =============================================================================
# MÓDULO DE FILTROS
# =============================================================================

class DataFilter:
    """Filtros para dados"""
    
    @staticmethod
    def get_product_column(df: pd.DataFrame, categoria: str) -> Optional[str]:
        """Identificar a coluna correta de produto baseada na categoria"""

        if categoria == 'comercializacao':
            if 'cultivar' in df.columns:
                return 'cultivar'
        
        if 'produto' in df.columns:
            return 'produto'
        elif 'Produto' in df.columns:
            return 'Produto'
        
        possible_columns = ['cultivar', 'produto', 'Produto', 'item', 'Item']
        for col in possible_columns:
            if col in df.columns:
                return col
        
        return None
    
    @staticmethod
    def filter_data(df: pd.DataFrame, filters: FilterParams, categoria: str) -> pd.DataFrame:
        """Aplicar filtros aos dados"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        if EmbrapaDataProcessor.is_import_export_category(categoria):
            if filters.pais:
                # Procurar coluna de país (pode ter nomes diferentes)
                country_columns = [col for col in df.columns if 'país' in col.lower() or 'pais' in col.lower()]
                if country_columns:
                    country_col = country_columns[0]
                    filtered_df = filtered_df[
                        filtered_df[country_col].str.contains(filters.pais, case=False, na=False)
                    ]
            
            # Filtrar por intervalo de anos para importação/exportação
            if filters.ano_inicio or filters.ano_fim:
                year_columns = []
                for col in df.columns:
                    if '_quantidade_kg' in col or '_valor_usd' in col:
                        year_match = re.match(r'^(\d{4})_', col)
                        if year_match:
                            year_columns.append(col)
                
                if year_columns:
                    start_year = filters.ano_inicio or 1970
                    end_year = filters.ano_fim or 2024
                    
                    # Manter apenas colunas de anos no intervalo
                    cols_to_keep = []
                    for col in df.columns:
                        if '_quantidade_kg' in col or '_valor_usd' in col:
                            year_match = re.match(r'^(\d{4})_', col)
                            if year_match:
                                year = int(year_match.group(1))
                                if start_year <= year <= end_year:
                                    cols_to_keep.append(col)
                        else:
                            cols_to_keep.append(col)
                    
                    filtered_df = filtered_df[cols_to_keep]
        
        else:
            if filters.produto:
                product_column = DataFilter.get_product_column(df, categoria)
                if product_column:
                    filtered_df = filtered_df[
                        filtered_df[product_column].str.contains(filters.produto, case=False, na=False)
                    ]
            
            # Filtrar por intervalo de anos
            year_columns = [col for col in df.columns if col.isdigit()]
            if filters.ano_inicio or filters.ano_fim:
                if year_columns:
                    start_year = filters.ano_inicio or min([int(col) for col in year_columns])
                    end_year = filters.ano_fim or max([int(col) for col in year_columns])
                    
                    years_to_keep = [col for col in year_columns 
                                   if start_year <= int(col) <= end_year]
                    
                    base_columns = [col for col in df.columns if not col.isdigit()]
                    filtered_df = filtered_df[base_columns + years_to_keep]
        
        return filtered_df

# =============================================================================
# CONFIGURAÇÃO DA API
# =============================================================================

app = FastAPI(
    title="API Vitibrasil Embrapa",
    description="API para consulta de dados de produção, processamento, comercialização, importação e exportação de vinhos e derivados do site da Embrapa",
    version="1.0.0",
    contact={
        "name": "API Vitibrasil",
        "email": "contato@exemplo.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_manager = AuthManager()
security = HTTPBearer()

@lru_cache(maxsize=32)
def fetch_embrapa_data(categoria: str) -> pd.DataFrame:
    """Wrapper com cache para buscar dados"""
    return EmbrapaDataProcessor.fetch_and_process_data(categoria)

# =============================================================================
# FUNÇÕES DE DEPENDÊNCIA
# =============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obter usuário atual através do token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    if token_data.username is None:
        raise credentials_exception
    
    user = auth_manager.get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# =============================================================================
# ROTAS DA API
# =============================================================================

@app.get("/", tags=["Geral"])
async def root():
    """
    ✅ Rota raiz da API

    Fornece informações gerais sobre a API, incluindo:
    - Mensagem de boas-vindas
    - Versão da API
    - Endpoints úteis (/token, /docs, /categorias)
    """
    return {
        "message": "API Vitibrasil Embrapa",
        "description": "API para consulta de dados de vitivinicultura do RS",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/token",
            "docs": "/docs",
            "categorias": list(EMBRAPA_URLS.keys())
        }
    }

@app.get("/categorias", tags=["Geral"])
async def listar_categorias():
    """
    📂 Listar categorias de dados disponíveis

    Retorna:
    - Lista com todas as categorias suportadas
    - Descrição de cada categoria
    - Agrupamentos por tipo (produção, importação etc.)
    """
    return {
        "categorias": list(EMBRAPA_URLS.keys()),
        "total_categorias": len(EMBRAPA_URLS),
        "descricoes": {
            "producao": "Quantidade de uvas processadas no RS",
            "processamento_viniferas": "Processamento de uvas viníferas",
            "processamento_americanas": "Processamento de uvas americanas e híbridas",
            "processamento_mesa": "Processamento de uvas de mesa",
            "processamento_semclass": "Processamento de uvas sem classificação",
            "comercializacao": "Comercialização de vinhos e derivados no RS",
            "importacao_vinhos": "Importação de vinhos de mesa",
            "importacao_espumantes": "Importação de espumantes",
            "importacao_frescas": "Importação de uvas frescas",
            "importacao_passas": "Importação de uvas passas",
            "importacao_suco": "Importação de suco de uva",
            "exportacao_vinhos": "Exportação de vinhos de mesa",
            "exportacao_espumantes": "Exportação de espumantes",
            "exportacao_uva": "Exportação de uvas frescas",
            "exportacao_sucos": "Exportação de suco de uva"
        },
        "agrupamentos": {
            "producao": ["producao"],
            "processamento": [
                "processamento_viniferas", 
                "processamento_americanas", 
                "processamento_mesa", 
                "processamento_semclass"
            ],
            "comercializacao": ["comercializacao"],
            "importacao": [
                "importacao_vinhos", 
                "importacao_espumantes", 
                "importacao_frescas", 
                "importacao_passas", 
                "importacao_suco"
            ],
            "exportacao": [
                "exportacao_vinhos", 
                "exportacao_espumantes", 
                "exportacao_uva", 
                "exportacao_sucos"
            ]
        }
    }

@app.post("/token", response_model=Token, tags=["Autenticação"])
async def gerar_token_de_acesso(username: str, password: str):
    """
    🔐 Obter token de acesso (JWT)

    *Obs: A API ainda não conta com banco de dados, utilizar o usuário ADMIN encontrado nos parâmetros de exemplo.

    Parâmetros:
    - username: admin
    - password: secret

    Retorna:
    - Token JWT para autenticação nas rotas protegidas
    """
    user = auth_manager.authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nome de usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/dados/{categoria}", response_model=DataResponse, tags=["Dados"])
async def obter_dados(
    categoria: str,
    produto: Optional[str] = None,
    pais: Optional[str] = None,
    ano_inicio: Optional[int] = None,
    ano_fim: Optional[int] = None,
    limit: Optional[int] = 100,
    current_user: User = Depends(get_current_user)
):
    """
    📊 Consultar dados de uma categoria (autenticado)

    Parâmetros:
    - categoria: Nome da categoria (ex: 'producao', 'importacao_vinhos')
    - produto: Filtrar por nome do produto
    - pais: Filtrar por país (apenas em importação/exportação)
    - ano_inicio: Filtro de ano inicial
    - ano_fim: Filtro de ano final
    - limit: Limite máximo de registros (padrão: 100)

    Retorna:
    - Dados filtrados e metadados da consulta
    """
    # Buscar dados da categoria
    df = fetch_embrapa_data(categoria)
    
    if df.empty:
        return DataResponse(
            categoria=categoria,
            total_registros=0,
            dados=[],
            metadata={
                "filtros_aplicados": {
                    "produto": produto,
                    "pais": pais,
                    "ano_inicio": ano_inicio,
                    "ano_fim": ano_fim,
                    "limit": limit
                },
                "mensagem": "Nenhum registro encontrado para a categoria especificada"
            }
        )
    
    # Aplicar filtros
    filters = FilterParams(
        produto=produto,
        pais=pais,
        ano_inicio=ano_inicio,
        ano_fim=ano_fim,
        limit=limit
    )
    
    filtered_df = DataFilter.filter_data(df, filters, categoria)
    
    # Aplicar limite de registros
    if limit and limit > 0:
        filtered_df = filtered_df.head(limit)
    
    # Converter para lista de dicionários com conversão explícita de tipos
    dados_raw = filtered_df.to_dict('records')
    dados = []
    
    for registro in dados_raw:
        # Converter cada registro para Dict[str, Any]
        registro_convertido: Dict[str, Any] = {}
        for chave, valor in registro.items():
            # Garantir que a chave seja string
            chave_str = str(chave)
            # Converter valores pandas/numpy para tipos Python nativos
            if pd.isna(valor):
                registro_convertido[chave_str] = None
            elif hasattr(valor, 'item'):  # numpy types
                registro_convertido[chave_str] = valor.item()
            else:
                registro_convertido[chave_str] = valor
        dados.append(registro_convertido)
    
    # Preparar metadata
    metadata = {
        "categoria": categoria,
        "total_registros_categoria": len(df),
        "registros_apos_filtros": len(filtered_df),
        "colunas": list(filtered_df.columns),
        "filtros_aplicados": {
            "produto": produto,
            "pais": pais,
            "ano_inicio": ano_inicio,
            "ano_fim": ano_fim,
            "limit": limit
        },
        "tipo_categoria": "importacao_exportacao" if EmbrapaDataProcessor.is_import_export_category(categoria) else "producao_processamento"
    }
    
    return DataResponse(
        categoria=categoria,
        total_registros=len(filtered_df),
        dados=dados,
        metadata=metadata
    )

@app.get("/dados/{categoria}/produtos", tags=["Dados"])
async def listar_produtos(
    categoria: str,
    current_user: User = Depends(get_current_user)
):
    """
    📦 Listar produtos ou países por categoria (autenticado)

    Se a categoria for de importação/exportação, retorna os países.
    Caso contrário, retorna a lista de produtos disponíveis.

    Parâmetros:
    - categoria: Nome da categoria

    Retorna:
    - Lista de produtos ou países
    """
    df = fetch_embrapa_data(categoria)
    
    if df.empty:
        return {
            "categoria": categoria,
            "mensagem": "Nenhum registro encontrado",
            "total_itens": 0,
            "itens": []
        }
    
    # Para importação/exportação, listar países
    if EmbrapaDataProcessor.is_import_export_category(categoria):
        # Procurar coluna de país
        country_columns = [col for col in df.columns if 'país' in col.lower() or 'pais' in col.lower()]
        
        if country_columns:
            country_col = country_columns[0]
            paises = df[country_col].dropna().unique().tolist()
            paises = [pais for pais in paises if str(pais).strip() != '']
            paises.sort()
            
            return {
                "categoria": categoria,
                "tipo": "paises",
                "total_itens": len(paises),
                "itens": paises
            }
        else:
            return {
                "categoria": categoria,
                "tipo": "paises",
                "mensagem": "Coluna de países não encontrada",
                "total_itens": 0,
                "itens": []
            }
    
    else:
        product_column = DataFilter.get_product_column(df, categoria)
        
        if product_column:
            produtos = df[product_column].dropna().unique().tolist()
            produtos = [produto for produto in produtos if str(produto).strip() != '']
            produtos.sort()
            
            return {
                "categoria": categoria,
                "tipo": "produtos",
                "coluna_produto": product_column,
                "total_itens": len(produtos),
                "itens": produtos
            }
        else:
            return {
                "categoria": categoria,
                "tipo": "produtos",
                "mensagem": "Coluna de produtos não encontrada",
                "total_itens": 0,
                "itens": []
            }

@app.get("/dados/{categoria}/anos", tags=["Dados"])
async def listar_anos(
    categoria: str,
    current_user: User = Depends(get_current_user)
):
    """
    📅 Listar anos disponíveis na categoria (autenticado)

    Retorna:
    - Lista de anos presentes nos dados da categoria
    - Ano inicial e final disponíveis

    Parâmetros:
    - categoria: Nome da categoria
    """
    df = fetch_embrapa_data(categoria)
    
    if df.empty:
        return {
            "categoria": categoria,
            "mensagem": "Nenhum registro encontrado",
            "total_anos": 0,
            "anos": []
        }
    
    anos = []
    
    # Para importação/exportação, extrair anos das colunas especiais
    if EmbrapaDataProcessor.is_import_export_category(categoria):
        for col in df.columns:
            if '_quantidade_kg' in col or '_valor_usd' in col:
                year_match = re.match(r'^(\d{4})_', col)
                if year_match:
                    year = int(year_match.group(1))
                    if year not in anos:
                        anos.append(year)
    else:
        # Para outras categorias, procurar colunas que são anos
        year_columns = [col for col in df.columns if col.isdigit()]
        anos = [int(year) for year in year_columns]
    
    anos = sorted(anos)
    
    return {
        "categoria": categoria,
        "ano_inicial": min(anos) if anos else None,
        "ano_final": max(anos) if anos else None,
        "total_anos": len(anos),
        "anos": anos
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )