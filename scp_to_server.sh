
# scp -i ~/.ssh/dm_deep_learning_kp.pem ${1} ubuntu@ec2-${IP}.eu-west-2.compute.amazonaws.com:~

scp -r -i ~/.ssh/dm_deep_learning_kp.pem ${1} ec2-user@ec2-${IP}.eu-west-2.compute.amazonaws.com:~/