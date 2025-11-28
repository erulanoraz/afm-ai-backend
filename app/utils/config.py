# app/utils/config.py
import os
from typing import Dict
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """🔧 Конфигурация приложения AFM Legal AI"""

    # ==========================================
    # PostgreSQL
    # ==========================================
    DB_HOST: str = Field(default="localhost", description="Хост PostgreSQL")
    DB_PORT: str = Field(default="5432", description="Порт PostgreSQL")
    DB_NAME: str = Field(default="afm", description="Имя БД")
    DB_USER: str = Field(default="afm", description="Пользователь БД")
    DB_PASS: str = Field(default="afm_password", description="Пароль БД")
    
    # ==========================================
    # S3 / MinIO
    # ==========================================
    S3_ENDPOINT: str = Field(default="http://localhost:9000", description="S3 endpoint")
    S3_ACCESS_KEY: str = Field(default="minioadmin", description="S3 access key")
    S3_SECRET_KEY: str = Field(default="minioadmin", description="S3 secret key")
    S3_BUCKET: str = Field(default="afm-originals", description="S3 bucket name")
    S3_REGION: str = Field(default="us-east-1", description="S3 region")
    
    # ==========================================
    # Приложение
    # ==========================================
    APP_NAME: str = Field(default="AFM Legal AI", description="Название приложения")
    APP_VERSION: str = Field(default="1.0.0", description="Версия приложения")
    APP_ENV: str = Field(default="dev", description="Окружение: dev/staging/prod")
    DEBUG: bool = Field(default=False, description="Режим отладки")
    LOG_LEVEL: str = Field(default="INFO", description="Уровень логирования")
    
    # ==========================================
    # Обработка документов
    # ==========================================
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Максимальный размер файла (MB)")
    MIN_DOC_LENGTH: int = Field(default=1, description="Минимальная длина документа")
    CHUNK_TOKENS: int = Field(default=400, description="Размер чанка в токенах")
    CHUNK_OVERLAP: int = Field(default=80, description="Перекрытие чанков")
    ALLOWED_FILE_TYPES: list = Field(
        default=[".pdf", ".docx", ".doc", ".txt"],
        description="Допустимые типы файлов"
    )
    
    # ==========================================
    # Пути к OCR и Poppler
    # ==========================================
    TESSERACT_PATH: str = Field(
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        description="Путь к Tesseract OCR"
    )

    POPPLER_PATH: str = Field(
        default=r"C:\Users\User\Desktop\poppler-25.07.0\Library\bin",
        description="Путь к Poppler (pdfinfo/pdftoppm)"
    )

    # ==========================================
    # LLM / AI Qualifier
    # ==========================================
    LLM_API_URL: str = Field(
        default="http://92.46.59.74:8000/v1/chat/completions",
        description="URL LLM API"
    )
    LLM_API_KEY: str = Field(default="local", description="API ключ для LLM")
    LLM_MODEL: str = Field(default="gpt-oss-120b", description="Модель LLM")
    LLM_TEMPERATURE: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Температура генерации (0.0-2.0)"
    )
    LLM_TIMEOUT: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Таймаут LLM запроса (сек)"
    )
    MAX_RETRY_ATTEMPTS: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Количество повторных попыток"
    )

    
    # ==========================================
    # Гибридный поиск
    # ==========================================
    W_SEM: float = Field(default=0.6, ge=0.0, le=1.0, description="Вес семантического поиска")
    W_KW: float = Field(default=0.3, ge=0.0, le=1.0, description="Вес ключевых слов")
    W_META: float = Field(default=0.1, ge=0.0, le=1.0, description="Вес метаданных")
    
    TOPK_VECTOR: int = Field(default=100, ge=1, description="Top-K для векторного поиска")
    TOPK_BM25: int = Field(default=100, ge=1, description="Top-K для BM25")
    TOPN_RERANK: int = Field(default=20, ge=1, description="Top-N для реранкинга")
    TOPN_FINAL: int = Field(default=25, ge=1, description="Финальное количество результатов")
    
    # ==========================================
    # OpenSearch / Elasticsearch
    # ==========================================
    ES_URL: str = Field(default="http://localhost:9200", description="OpenSearch URL")
    ES_USER: str = Field(default="admin", description="OpenSearch пользователь")
    ES_PASS: str = Field(default="admin", description="OpenSearch пароль")
    ES_INDEX_PREFIX: str = Field(default="afm_legal", description="Префикс индексов")
    ES_TIMEOUT: int = Field(default=30, description="Таймаут OpenSearch (сек)")
    

    # ==========================================
    # Celery / Redis
    # ==========================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL для Celery брокера"
    )


    # ==========================================
    # Weaviate / Embeddings
    # ==========================================
    WEAVIATE_URL: str = Field(
        default="http://localhost:8080",
        description="Weaviate endpoint"
    )

    LLM_EMBEDDING_URL: str = Field(
        default="http://localhost:8080/v1/embeddings",
        description="Embedding endpoint (Weaviate)"
    )


    # ==========================================
    # Anti-hallucination / Verification
    # ==========================================
    CONF_THRESH_CRITICAL: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Порог уверенности для критических фактов"
    )
    CONF_THRESH_DEFAULT: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Порог уверенности по умолчанию"
    )
    REQUIRE_TWO_SOURCES: bool = Field(
        default=True,
        description="Требовать минимум 2 источника для фактов"
    )
    ENFORCE_CITATIONS: bool = Field(
        default=True,
        description="Строгая проверка цитирований"
    )
    RETURN_INSUFFICIENT_ON_FAIL: bool = Field(
        default=True,
        description="Возвращать 'недостаточно данных' при провале верификации"
    )
    
    # ==========================================
    # Validation
    # ==========================================
    @field_validator('APP_ENV')
    @classmethod
    def validate_env(cls, v):
        """Проверка допустимых окружений"""
        allowed = ['dev', 'development', 'staging', 'prod', 'production']
        if v.lower() not in allowed:
            raise ValueError(f"APP_ENV должен быть одним из: {allowed}")
        return v.lower()
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        """Проверка допустимых уровней логирования"""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL должен быть одним из: {allowed}")
        return v_upper
    
    # ==========================================
    # Computed properties
    # ==========================================
    @property
    def DB_URL(self) -> str:
        """Формирование URL подключения к PostgreSQL"""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def HYBRID_WEIGHTS(self) -> Dict[str, float]:
        """Веса для гибридного поиска"""
        return {
            "w_sem": self.W_SEM,
            "w_kw": self.W_KW,
            "w_meta": self.W_META,
        }
    
    @property
    def WEIGHTS_SUM(self) -> float:
        """Сумма всех весов (для проверки)"""
        return self.W_SEM + self.W_KW + self.W_META
    
    @property
    def IS_PRODUCTION(self) -> bool:
        """Проверка продакшн-окружения"""
        return self.APP_ENV in ['prod', 'production']
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        """Проверка dev-окружения"""
        return self.APP_ENV in ['dev', 'development']
    
    @property
    def S3_CONFIG(self) -> Dict[str, str]:
        """Конфигурация S3 для boto3"""
        return {
            "endpoint_url": self.S3_ENDPOINT,
            "aws_access_key_id": self.S3_ACCESS_KEY,
            "aws_secret_access_key": self.S3_SECRET_KEY,
            "region_name": self.S3_REGION,
        }
    
    @property
    def ES_CONFIG(self) -> Dict[str, str]:
        """Конфигурация OpenSearch"""
        return {
            "hosts": [self.ES_URL],
            "http_auth": (self.ES_USER, self.ES_PASS),
            "timeout": self.ES_TIMEOUT,
            "use_ssl": self.ES_URL.startswith("https"),
            "verify_certs": self.IS_PRODUCTION,
        }
    
    # ==========================================
    # Pydantic v2 Config
    # ==========================================
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",  # ✅ UTF-8 support
        case_sensitive=True,
        extra="allow",  # ✅ Разрешить дополнительные поля из .env
        validate_default=True,
        validate_assignment=True,
    )
    
    # ==========================================
    # Post-init validation
    # ==========================================
    def model_post_init(self, __context) -> None:
        """Дополнительная валидация после инициализации"""
        # Проверка суммы весов
        weights_sum = self.WEIGHTS_SUM
        if not (0.99 <= weights_sum <= 1.01):
            raise ValueError(
                f"Сумма весов должна быть ≈1.0, получено: {weights_sum:.3f} "
                f"(W_SEM={self.W_SEM}, W_KW={self.W_KW}, W_META={self.W_META})"
            )
        
        # Предупреждение о секретах в dev-режиме
        if self.IS_DEVELOPMENT and (
            self.DB_PASS == "afm_password" or
            self.LLM_API_KEY == "local" or
            self.S3_SECRET_KEY == "minioadmin"
        ):
            import warnings
            warnings.warn(
                "⚠️  Используются дефолтные пароли в dev-режиме! "
                "Для продакшена установите безопасные значения в .env",
                UserWarning
            )


# ==========================================
# Глобальный экземпляр
# ==========================================
settings = Settings()


# ==========================================
# Вспомогательные функции
# ==========================================
def get_db_url(echo: bool = False) -> str:
    """
    Получить URL БД с опциональным логированием SQL
    
    Args:
        echo: Выводить SQL-запросы в консоль
    
    Returns:
        str: URL подключения
    """
    url = settings.DB_URL
    if echo:
        url += "?echo=true"
    return url


def validate_config() -> Dict[str, bool]:
    """
    Проверка доступности всех сервисов
    
    Returns:
        Dict с результатами проверки каждого компонента
    """
    results = {
        "database": False,
        "llm_api": False,
        "opensearch": False,
        "s3": False,
    }
    
    # 1. Проверка БД
    try:
        from sqlalchemy import create_engine
        engine = create_engine(settings.DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        results["database"] = True
        engine.dispose()
    except Exception as e:
        print(f"❌ БД недоступна: {e}")
    
    # 2. Проверка LLM API
    try:
        import requests
        health_url = settings.LLM_API_URL.replace("/v1/chat/completions", "/health")
        resp = requests.get(health_url, timeout=5)
        results["llm_api"] = resp.status_code < 500
    except Exception as e:
        print(f"⚠️  LLM API недоступен: {e}")
    
    # 3. Проверка OpenSearch
    try:
        import requests
        from requests.auth import HTTPBasicAuth
        resp = requests.get(
            f"{settings.ES_URL}/_cluster/health",
            auth=HTTPBasicAuth(settings.ES_USER, settings.ES_PASS),
            timeout=5,
            verify=False
        )
        results["opensearch"] = resp.status_code == 200
    except Exception as e:
        print(f"⚠️  OpenSearch недоступен: {e}")
    
    # 4. Проверка S3/MinIO
    try:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client('s3', **settings.S3_CONFIG)
        s3.head_bucket(Bucket=settings.S3_BUCKET)
        results["s3"] = True
    except Exception as e:
        print(f"⚠️  S3/MinIO недоступен: {e}")
    
    return results


def print_config_summary():
    """Красивый вывод конфигурации"""
    print("=" * 70)
    print("🔧 AFM Legal AI - Конфигурация")
    print("=" * 70)
    
    print(f"\n📦 Приложение:")
    print(f"   • Название: {settings.APP_NAME}")
    print(f"   • Версия: {settings.APP_VERSION}")
    print(f"   • Окружение: {settings.APP_ENV}")
    print(f"   • Debug: {settings.DEBUG}")
    print(f"   • Log Level: {settings.LOG_LEVEL}")
    
    print(f"\n🗄️  База данных:")
    # Скрываем пароль
    safe_url = settings.DB_URL.replace(f":{settings.DB_PASS}@", ":****@")
    print(f"   • URL: {safe_url}")
    print(f"   • Host: {settings.DB_HOST}:{settings.DB_PORT}")
    print(f"   • Database: {settings.DB_NAME}")
    print(f"   • User: {settings.DB_USER}")
    
    print(f"\n☁️  S3 / MinIO:")
    print(f"   • Endpoint: {settings.S3_ENDPOINT}")
    print(f"   • Bucket: {settings.S3_BUCKET}")
    print(f"   • Region: {settings.S3_REGION}")
    print(f"   • Access Key: {settings.S3_ACCESS_KEY[:4]}****")
    
    print(f"\n🤖 LLM API:")
    print(f"   • Endpoint: {settings.LLM_API_URL}")
    print(f"   • Model: {settings.LLM_MODEL}")
    print(f"   • Temperature: {settings.LLM_TEMPERATURE}")
    print(f"   • Timeout: {settings.LLM_TIMEOUT}s")
    print(f"   • Max Retries: {settings.MAX_RETRY_ATTEMPTS}")
    
    print(f"\n🔍 Гибридный поиск:")
    print(f"   • Веса: SEM={settings.W_SEM}, KW={settings.W_KW}, META={settings.W_META}")
    print(f"   • Сумма весов: {settings.WEIGHTS_SUM:.3f}")
    print(f"   • Top-K: vector={settings.TOPK_VECTOR}, BM25={settings.TOPK_BM25}")
    print(f"   • Rerank: {settings.TOPN_RERANK}, Final: {settings.TOPN_FINAL}")
    
    print(f"\n🔎 OpenSearch / Elasticsearch:")
    print(f"   • URL: {settings.ES_URL}")
    print(f"   • Index Prefix: {settings.ES_INDEX_PREFIX}")
    print(f"   • Timeout: {settings.ES_TIMEOUT}s")
    print(f"   • User: {settings.ES_USER}")
    
    print(f"\n✅ Верификация (Anti-hallucination):")
    print(f"   • Порог (критический): {settings.CONF_THRESH_CRITICAL}")
    print(f"   • Порог (обычный): {settings.CONF_THRESH_DEFAULT}")
    print(f"   • Требовать 2 источника: {settings.REQUIRE_TWO_SOURCES}")
    print(f"   • Строгие цитаты: {settings.ENFORCE_CITATIONS}")
    print(f"   • Возврат 'недостаточно': {settings.RETURN_INSUFFICIENT_ON_FAIL}")
    
    print(f"\n📂 Обработка файлов:")
    print(f"   • Max размер: {settings.MAX_FILE_SIZE_MB} MB")
    print(f"   • Min длина: {settings.MIN_DOC_LENGTH} символов")
    print(f"   • Типы: {', '.join(settings.ALLOWED_FILE_TYPES)}")
    print(f"   • Chunk размер: {settings.CHUNK_TOKENS} токенов")
    print(f"   • Chunk overlap: {settings.CHUNK_OVERLAP} токенов")
    
    print("\n" + "=" * 70)


# ==========================================
# Тест при прямом запуске
# ==========================================
if __name__ == "__main__":
    print_config_summary()
    
    print("\n🔍 Проверка доступности сервисов...")
    results = validate_config()
    
    print("\n📊 Результаты проверки:")
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {service.upper()}: {'OK' if status else 'НЕДОСТУПЕН'}")
    
    all_ok = all(results.values())
    if all_ok:
        print("\n✅ Все сервисы доступны!")
    else:
        print("\n⚠️  Некоторые сервисы недоступны. Проверьте конфигурацию.")
