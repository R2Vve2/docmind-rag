# Information Security Policy

## Scope

This policy applies to all Northwind Robotics employees, contractors and interns, and to every device that accesses company systems.

## Passwords and authentication

Passwords must be at least 14 characters long and must not be reused across systems. Multi-factor authentication (MFA) is mandatory for email, GitHub, AWS, Workday and the VPN. Hardware security keys are issued to engineers with production access and are the only accepted second factor for AWS root and administrator accounts.

Shared accounts are prohibited. Service accounts must be owned by a named team and have their credentials rotated every 90 days.

## Devices

Company laptops must have full-disk encryption enabled, automatic updates turned on and the CrowdStrike agent installed. Personal devices may access email and Slack only through the managed mobile apps. Screens must lock after five minutes of inactivity.

## Data classification

Data is classified as Public, Internal, Confidential or Restricted. Restricted data (customer robot telemetry, source code for the safety controller, payroll data) may only be stored in approved systems and must never be copied to personal cloud storage or sent to personal email addresses.

## Secrets management

API keys, certificates and database credentials must be stored in HashiCorp Vault. Committing secrets to a git repository is a security incident even if the repository is private. Pre-commit hooks running gitleaks are mandatory in every repository.

## Incident response

Suspected security incidents must be reported within one hour to security@northwind-robotics.example or the #security-incidents Slack channel. The on-call security engineer triages the report within 30 minutes during business hours and within two hours outside business hours. Severity 1 incidents (confirmed data breach or production compromise) trigger the incident bridge and require a written post-mortem within five working days.

## Access reviews

Managers review the access rights of their reports every quarter. Access for leavers must be revoked on their last working day; the People team notifies IT at least two working days in advance.

## Acceptable use

Company systems may be used for reasonable personal purposes provided it does not interfere with work or violate any law. Installing unlicensed software, disabling security controls or using company resources for cryptocurrency mining is prohibited.

## Training

All staff complete security awareness training within 30 days of joining and annually thereafter. Engineers additionally complete secure coding training every year.
