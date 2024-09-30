#!/bin/bash
set -e
set -u
set -o pipefail

# Redirect all output to a log file
exec > >(tee -i deploy.log)
exec 2>&1

# Trap errors and exit
trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

# Load environment variables from config.env
source config.env

# Variables
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPO_NAME=${ECR_REPO_NAME:-xai-loan-prediction-repo}
IMAGE_TAG=${IMAGE_TAG:-latest}
SERVICE_NAME=${SERVICE_NAME:-xai-loan-prediction-service}
CLUSTER_NAME=${CLUSTER_NAME:-xai-loan-prediction-cluster}

# Authenticate Docker to AWS ECR
echo "Authenticating Docker to AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query 'Account' --output text).dkr.ecr.$AWS_REGION.amazonaws.com"

# Build the Docker image
echo "Building Docker image..."
docker build -t $ECR_REPO_NAME:$IMAGE_TAG .

# Tag the Docker image for ECR
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

echo "Tagging Docker image with ECR URI: $ECR_URI:$IMAGE_TAG"
docker tag $ECR_REPO_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG

# Push the Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_URI:$IMAGE_TAG

# Update ECS service with the new image
echo "Updating ECS service..."
aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment

# Wait for the service to stabilize
echo "Waiting for ECS service to stabilize..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME
echo "ECS service is now stable."

# Optional: Remove local Docker images
echo "Cleaning up local Docker images..."
docker rmi $ECR_REPO_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG || true

echo "Deployment completed successfully."
