#!/usr/bin/env bash
set -ex
##
##
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.9 --unlabel_percent 0.9 --batch_size 8 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.9 --unlabel_percent 0.9
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.8 --unlabel_percent 0.8 --batch_size 8 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.8 --unlabel_percent 0.8
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.7 --unlabel_percent 0.7 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.7 --unlabel_percent 0.7
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.6 --unlabel_percent 0.6 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.6 --unlabel_percent 0.6
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.5 --unlabel_percent 0.5 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.5 --unlabel_percent 0.5
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.4 --unlabel_percent 0.4 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.4 --unlabel_percent 0.4
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.3 --unlabel_percent 0.3 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.3 --unlabel_percent 0.3
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.2 --unlabel_percent 0.2 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.2 --unlabel_percent 0.2
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.1 --unlabel_percent 0.1 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.1 --unlabel_percent 0.1
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
#python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.02 --unlabel_percent 0.02 --batch_size 24 #
#python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.02 --unlabel_percent 0.02
#echo "$(date "+%Y.%m.%d-%H.%M.%S")"
#
python main.py --mode train --checkpoint_path ./checkpoint/Semi-base-0.00 --unlabel_percent 0.0 --batch_size 16 #
python main.py --mode inference --checkpoint_path ./checkpoint/Semi-base-0.00 --unlabel_percent 0.0
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
