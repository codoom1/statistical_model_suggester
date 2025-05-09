"""Add resume_url and admin_notes to expert applications

Revision ID: 5c442376367c
Revises: 
Create Date: 2025-04-13 21:11:02.948339

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5c442376367c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('consultation')
    op.drop_table('user')
    op.drop_table('expert_application')
    op.drop_table('analysis')
    with op.batch_alter_table('expert_applications', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resume_url', sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column('admin_notes', sa.Text(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('expert_applications', schema=None) as batch_op:
        batch_op.drop_column('admin_notes')
        batch_op.drop_column('resume_url')

    op.create_table('analysis',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.Column('research_question', sa.VARCHAR(length=500), nullable=False),
    sa.Column('analysis_goal', sa.VARCHAR(length=50), nullable=True),
    sa.Column('dependent_variable', sa.VARCHAR(length=50), nullable=True),
    sa.Column('independent_variables', sa.TEXT(), nullable=True),
    sa.Column('sample_size', sa.VARCHAR(length=20), nullable=True),
    sa.Column('missing_data', sa.VARCHAR(length=50), nullable=True),
    sa.Column('data_distribution', sa.VARCHAR(length=50), nullable=True),
    sa.Column('relationship_type', sa.VARCHAR(length=50), nullable=True),
    sa.Column('recommended_model', sa.VARCHAR(length=100), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('expert_application',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.Column('email', sa.VARCHAR(length=120), nullable=False),
    sa.Column('areas_of_expertise', sa.TEXT(), nullable=False),
    sa.Column('institution', sa.VARCHAR(length=200), nullable=True),
    sa.Column('bio', sa.TEXT(), nullable=True),
    sa.Column('status', sa.VARCHAR(length=20), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('username', sa.VARCHAR(length=80), nullable=False),
    sa.Column('email', sa.VARCHAR(length=120), nullable=False),
    sa.Column('password_hash', sa.VARCHAR(length=128), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('_is_admin', sa.BOOLEAN(), nullable=True),
    sa.Column('_is_expert', sa.BOOLEAN(), nullable=True),
    sa.Column('is_approved_expert', sa.BOOLEAN(), nullable=True),
    sa.Column('areas_of_expertise', sa.TEXT(), nullable=True),
    sa.Column('institution', sa.VARCHAR(length=200), nullable=True),
    sa.Column('bio', sa.TEXT(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('consultation',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('requester_id', sa.INTEGER(), nullable=False),
    sa.Column('expert_id', sa.INTEGER(), nullable=True),
    sa.Column('analysis_id', sa.INTEGER(), nullable=True),
    sa.Column('title', sa.VARCHAR(length=200), nullable=False),
    sa.Column('description', sa.TEXT(), nullable=True),
    sa.Column('status', sa.VARCHAR(length=20), nullable=True),
    sa.Column('response', sa.TEXT(), nullable=True),
    sa.Column('is_public', sa.BOOLEAN(), nullable=True),
    sa.Column('analysis_goal', sa.VARCHAR(length=50), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['analysis_id'], ['analysis.id'], ),
    sa.ForeignKeyConstraint(['expert_id'], ['user.id'], ),
    sa.ForeignKeyConstraint(['requester_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###
