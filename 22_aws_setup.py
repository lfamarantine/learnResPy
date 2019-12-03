# AWS WORKINGS -----------------------------------------------
import boto3

# AWS Services = Home Utilities
# step 1: create aws account
# step 2: https://eu-west-2.console.aws.amazon.com/console/home?region=eu-west-2#
# step 3: search for IAM
# step 4: Ssers

# .. services: IAM, S3, SNS, Comprehend, Rekognition

# login ----------------------
aws_key = ""
aws_sec = ""
s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id=aws_key, aws_secret_access_key=aws_sec)
response = s3.list_buckets()

# aws buckets..
buckets = s3.list_buckets()

# multiple clients..
sns = boto3.client("sns", region_name="us-east-1", aws_access_key_id=aws_key, aws_secret_access_key=aws_sec)

# aws topics..
topics = sns.list_topics()

# BUCKETS ---------------------
# s3 let's us put any file into the cloud and manage it via a url
# main componenets of s3: objects (= files in folders)  & buckets (= folders on desktop)
# buckets: own permission policy, website storage, generate logs

# s3.create_bucket(Bucket="buck")
# s3.list_buckets()
# s3.delete_bucket(Bucket="buck")

# create 3 buckets..
response_staging = s3.create_bucket(Bucket='proj-staging')
response_processed = s3.create_bucket(Bucket='proj-processed')
response_test = s3.create_bucket(Bucket='proj-test')

# list the buckets..
response = s3.list_buckets()
for bucket in response['Buckets']:
    # Print the Name for each bucket
    print(bucket['Name'])

# upload files..
s3.upload_file(Filename="data/airquality.csv", Bucket="proj-staging", Key="airquality")
s3.list_objects(Bucket="proj-staging")
s3.download_file(Filename="data/airquality.csv", Bucket="proj-staging", Key="airquality")
s3.delete_object(Bucket="proj-staging", Key="airquality")




