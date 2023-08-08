import time
import boto3
import argparse
import sys, os

import logging
import logging.handlers

from time import gmtime, strftime


def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()


def get_approved_package(model_package_group_name, sm_client):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """

    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
        
        
        
#########################################
## 커맨드 인자 처리
#########################################

# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model_package_group_name', type=str)
parser.add_argument('--region', type=str, default="ap-northeast-2")
parser.add_argument('--endpoint_instance_type', type=str, default='ml.t3.medium')
parser.add_argument('--role', type=str)
args = parser.parse_args()

logger.info("#############################################")
logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
logger.info(f"args.region: {args.region}")    
logger.info(f"args.endpoint_instance_type: {args.endpoint_instance_type}")        
logger.info(f"args.role: {args.role}")    



region = args.region

boto3.setup_default_session(region_name=region)
sagemaker_boto_client = boto3.client('sagemaker')

# Get the latest approved package
model_package_arn = get_approved_package(args.model_package_group_name, sagemaker_boto_client)
model_name = model_package_arn.split('/')[-2] + "-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_name = model_package_arn.split('/')[-2] + "-latest"

#name truncated per sagameker length requirememnts (63 char max)
endpoint_config_name=f'{model_name}-config'
existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']

instance_type = args.endpoint_instance_type
#########################################
## model 객체 생성
#########################################

print("Model name : {}".format(model_name))
container_list = [{'ModelPackageName': model_package_arn}]

create_model_response = sagemaker_boto_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = args.role,
    Containers = container_list
)


#########################################
## endpoint_config 생성
#########################################

if not existing_configs:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            'InitialVariantWeight': 1,
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)['Endpoints']

#########################################
## endpoint 생성
#########################################

if not existing_endpoints:
    logger.info(f"Creating endpoint")        
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
else:
    logger.info(f"Updating endpoint")        
    create_endpoint_response = sagemaker_boto_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)         
    

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']


logger.info(f'Endpoint status is {endpoint_status}')    
while endpoint_status in ['Creating','Updating']:
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    logger.info(f'Endpoint status: {endpoint_status}')
    if endpoint_status == 'Creating':
        time.sleep(30)