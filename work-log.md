# Work Log

## [2025-11-04]

### Time Spent
- Hours: 2

### Tasks Completed
- [X] Week 1
- [X] Week 2
- [X] Week 3

### What Worked Well
- Following the instructions
- Using uv for Python Virtual Environments
- Installed Rancher Desktop

### Challenges / Blockers
- Installing and setting up Python on Windows and the packages
- Setting up Visual Studio Code and extensions

### Notes & Decisions
- 

---

# Work Log

## [2025-11-05]

### Time Spent
- Hours: 1

### Tasks Completed
- [X] Week 4

### What Worked Well
- Created the Digital Ocean Droplet
- Asking Perplexity on the recommendations on how to create it worked well
- Using ssh to connect to the droplet

### Challenges / Blockers
- Finding a Promo Code that worked for Digital Ocean
- Promo Code that worked: SHIPITFAST10

### Notes & Decisions
- Took screenshots of every step.
- Deleted my Digital Ocean Droplet. Ran for 4 hours.

---

# Work Log

## [2025-11-06]

### Time Spent
- Hours: 1

### Tasks Completed
- [X] Week 5
- [X] Week 6
- [ ] Week 7

### What Worked Well
- I finally found a simple hello world example for Runpod that worked!
- Tutorial URL: https://www.runpod.io/blog/runpod-serverless-hello-world

### Challenges / Blockers
- Had an issue setting the environment variable in Powershell. I need to check the name of the variable. I had apikey and it needed to be token.
- $env:RUNPOD_API_TOKEN = "your_actual_token_here"
- This is how you set the environment variable in Powershell
- To check the environment variable in powershell run this command: Write-Output $env:RUNPOD_API_TOKEN or echo $env:RUNPOD_API_TOKEN
- Had an issue with the Github Action File. Took a few iterations to get it working.

### Notes & Decisions
- Here is the docker image I used that worked: docker.io/matthersh/runpod-hello-world
- Just started adding CI and CD Pipeline using Github Actions