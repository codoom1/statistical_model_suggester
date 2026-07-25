"""Add server-side questionnaire drafts.

Revision ID: 8f4e3d2c1b0a
Revises: 5c442376367c
"""

from alembic import op
import sqlalchemy as sa


revision = "8f4e3d2c1b0a"
down_revision = "5c442376367c"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "questionnaire_drafts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_questionnaire_drafts_user_id",
        "questionnaire_drafts",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_questionnaire_drafts_updated_at",
        "questionnaire_drafts",
        ["updated_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_questionnaire_drafts_updated_at",
        table_name="questionnaire_drafts",
    )
    op.drop_index(
        "ix_questionnaire_drafts_user_id",
        table_name="questionnaire_drafts",
    )
    op.drop_table("questionnaire_drafts")
