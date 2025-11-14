"""remove unique from s3_key"""

from alembic import op
import sqlalchemy as sa


revision = 'db53150cc2c9'
down_revision = 'fc0855bf88a9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Удаляем unique constraint, если он существует
    try:
        op.drop_constraint(
            "files_s3_key_key",
            "files",
            type_="unique"
        )
    except Exception:
        # Если ограничения нет — просто пропускаем
        pass


def downgrade() -> None:
    # Восстановление unique (если нужно откатывать миграцию)
    op.create_unique_constraint(
        "files_s3_key_key",
        "files",
        ["s3_key"]
    )
