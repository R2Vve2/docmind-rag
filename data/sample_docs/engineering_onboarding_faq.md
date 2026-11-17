# Engineering Onboarding FAQ

**How do I get access to the code?**
Ask your onboarding buddy to add you to the `northwind-eng` GitHub organisation. Access is granted through the Okta group `eng-github`, which your manager approves in Okta. Expect this to take under one business day.

**Which branching model do we use?**
Trunk-based development. All work happens on short-lived feature branches that are merged into `main` through a pull request. Pull requests need at least one approving review and a green CI run. Branches older than two weeks are flagged by the stale-branch bot.

**How are deployments done?**
Every merge to `main` builds a Docker image, pushes it to Amazon ECR and deploys automatically to the staging environment. Production deployments are promoted from staging through the Argo CD dashboard and require approval from the service owner. We deploy to production on Sunday through Wednesday; Thursday deployments require an exception from the on-call lead.

**What is the on-call rotation?**
Each backend team runs a weekly on-call rotation managed in PagerDuty. The on-call engineer is paid an on-call allowance and takes a compensatory day off after any night with a page between midnight and 06:00.

**Which languages and tools do we use?**
Backend services are written in Python 3.12 with FastAPI, and robot firmware is written in C++17. Infrastructure is defined with Terraform. We use GitHub Actions for CI, Grafana and Prometheus for monitoring, and Sentry for error tracking.

**How do I set up my development environment?**
Run the bootstrap script from the `dev-tools` repository. It installs `uv`, Docker Desktop, pre-commit hooks and the AWS CLI, and configures SSO for the `northwind-dev` AWS account. The full setup usually takes about 40 minutes.

**How do I request cloud resources?**
Open a ticket in the Platform Jira project using the "Cloud resource request" template. The platform team reviews requests daily. Anything with an estimated cost above 500 USD per month also needs sign-off from the engineering director.

**Where is documentation kept?**
Architecture decision records live in the `adr` folder of each repository. Team runbooks and design documents are in Confluence under the Engineering space. Customer-facing API documentation is published from the `docs` repository to docs.northwind-robotics.example.

**Who do I ask for help?**
Your onboarding buddy is your first point of contact for the first 30 days. The `#eng-help` Slack channel is monitored by all senior engineers, and the platform team can be reached in `#platform-support`.
