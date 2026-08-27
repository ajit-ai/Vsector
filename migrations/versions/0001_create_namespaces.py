"""create vsector_namespaces

Revision ID: 0001
Revises:
Create Date: 2026-08-27
"""

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    from alembic import op
    import sqlalchemy as sa

    op.create_table(
        "vsector_namespaces",
        sa.Column("name", sa.Text, primary_key=True),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

def downgrade():
    from alembic import op

    op.drop_table("vsector_namespaces")
