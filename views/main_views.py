from flask import Blueprint

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def AI_Model_run():
    return 'i dont know'