from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.utils.config import settings

# Создаём движок
engine = create_engine(settings.DB_URL, echo=False, future=True)

# Сессия
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Базовый класс для моделей
Base = declarative_base()


# Зависимость для FastAPI (используется в эндпоинтах)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Автоматическое создание таблиц при запуске
def init_db():
    print("[INIT] Creating database tables (if not exist)...")
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("[✓] Database initialized successfully")
