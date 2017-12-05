nohup python generator.py --config=../conf/train.conf --type=train --job_name=ps > log.ps &
nohup python generator.py --config=../conf/train.conf --type=train --job_name=worker --task_id=0 > log.0 &
nohup python generator.py --config=../conf/train.conf --type=train --job_name=worker --task_id=1 > log.1 &
