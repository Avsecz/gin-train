import argh
from gin_train.cli.gin_train import gin_train

# logging
import pkg_resources
import logging
import logging.config
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


def main():
    # assembling:
    # parser = argh.ArghParser()
    # parser.add_commands([gin_train])
    # argh.dispatch(parser)
    argh.dispatch_command(gin_train)
