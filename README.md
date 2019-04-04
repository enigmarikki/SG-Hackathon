# Sociate Generale Brain Waves 2019 Hackathon

Complaint Status Tracking

Problem Statement
Societe Generale (SocGen) is a French multinational banking and financial services company. With over 1,54,000 employees, based in 76 countries, they handle over 32 million clients throughout the world on a daily basis.

They provide services like retail banking, corporate and investment banking, asset management, portfolio management, insurance and other financial services.

While handling customer complaints, it is hard to track the status of the complaint. To automate this process, SocGen wants you to build a model that can automatically predict the complaint status (how the complaint was resolved) based on the complaint submitted by the consumer and other related meta-data.

Data Description
The dataset consists of three files: train.csv, test.csv and sample_submission.csv.

Column	Description
Complaint-ID	Complaint Id
Date received	Date on which the complaint was received
Transaction-Type	Type of transaction involved
Complaint-reason	Reason of the complaint
Consumer-complaint-summary	Complaint filed by the consumer - Present in three languages : English, Spanish, French
Company-response	Public response provided by the company (if any)
Date-sent-to-company	Date on which the complaint was sent to the respective department
Complaint-Status	Status of the complaint (Target Variable)
Consumer-disputes	If the consumer raised any disputes
Submission Format
Please submit the prediction as a .csv file in the format described below and in the sample submission file.

Complaint-ID	Complaint-Status
Te-1	Closed with explanation
Te-2	Closed with explanation
Te-3	Closed with explanation
Te-4	Closed with non-monetary relief
Te-5	Closed with explanation
Evaluation
The submissions will be evaluated on the f1 score with ‘weighted’ average.
