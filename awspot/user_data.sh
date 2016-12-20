#!/bin/bash
# This submitted via the ec2 configure instance page, additional options.  Node the output on the instance goes to /var/log/cloud-init-output.log
# Fill in the following after you've clone from our dummy snapshot.
# This will run as root account
# Designed to run with P2.xlarge (Nvidia K80, 12 GB card, good for running imagnet, inception..etc)


# 1. Configure all these for your specific case.  Docker is optional.  only us-west-2 and us-east-1 have p2 instances.
# Your image should be based of nvidia/cuda which is installed
TASK_PATH=<path to task on docker or local>
DOCKER_IMAGE=<docker_tag>
VOLUME_ID=<aws_volume_id>
AWS_ACCESS_KEY_ID=<XXXXX>
AWS_SECRET_ACCESS_KEY=<YYYYY>
REGION=us-west-2

# 2. This will attach the volume and resize it
INSTANCE_ID=`ec2metadata --instance-id`
aws --region=${REGION} ec2 attach-volume --instance-id ${INSTANCE_ID} --volume-id ${VOLUME_ID} --dev /dev/sdb
aws --region=${REGION} ec2 describe-volumes --volume-ids ${VOLUME_ID}
aws --region=${REGION} ec2 describe-volumes --volume-ids ${VOLUME_ID} | grep '"State": "attached"'
until aws --region=us-west-2 ec2 describe-volumes --volume-ids ${VOLUME_ID} | grep '"State": "attached"'; do
  sleep 1
  done
sudo resize2fs /dev/xvdb

# 3. mount the volume
mkdir -p /mnt/data && chown -R ubuntu:ubuntu /mnt/data
mount /dev/xvdb /mnt/data
mkdir -p /var/log/mylogs


# 4. Now run the task, it could be on the AMI, or you could download a package or git or anything u want
# Here we use docker.
# Pull the docker image.  Mount docker to the EBS drive.  Start task on nvidia docker.
docker pull ${DOCKER_IMAGE}
nvidia-docker run -v /mnt/data:/mnt/data -i ${DOCKER_IMAGE} bash -c "${TASK_PATH}" > /var/log/mylogs/docker-fractal.log 2>&1
