import os
import logging
import time
import yaml

__author__ = 'jeffye'


def setup_logging(
        default_path='config/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        add_time_stamp=False
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        if add_time_stamp:
            add_time_2_log_filename(config)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def add_time_2_log_filename(config):
    for k, v in config.iteritems():
        if k == 'filename':
            config[k] = v + "." + time.strftime("%Y-%d-%m-%s")
            print('log file name: %s' % config[k])
        elif type(v) is dict:
            add_time_2_log_filename(v)


def goal_prompt(logger, prompt='What are you testing in this experiment? '):
    print("            ***************************")
    goal = raw_input(prompt)
    logger.info("            ***************************")
    logger.info("TEST GOAL: %s" % goal)


def log_git_commit(logger):
    try:
        commit = get_git_revision_hash()
        logger.info("current git commit: %s" % commit)
    except:
        logger.info('cannot get git commit.')


def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
