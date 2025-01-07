## How to Set up Git Hub using Juypter

Very brief overview of how I did this - feel free to reachout if there are questions

---


### Making a repository:
- Green New Button on repositories page of github
- Name the repo, set to public, add a README
- Add gitignore to python (removes excess files that python creates)
- Press Create Repository

### Cloning the Repository:
- Click on the green code box in the code tab of the newly created repository
- Copy the HTTPs address there
- Go to juypter labs (or connected to the login server)
- Open a terminal in the location you would like to house the repository
- Run the command: git clone https://github.com/GitHubUser/RepoName
- This has cloned the repository to your jasmin user files

### Basic Git functions - use the internet for much better tutorials
- git status: tells you what you the state of changes within the repository, if you are ahead or behind on commits
- git add FILENAME: adds a specific file to the repository
- git add .: updates all the files that have been changed
- git commit -m message: commits to the branch
- git push: pusheds committed changes to github

### Issues I've run into and worked around
##### git push issues
- git push seems to not work with the standard password used for github
- To get around this I created a token to act as a password
- To do this go to github -> settings -> developer settings -> personal access tokens -> tokens (classic) -> generate new token -> generate new token (classic) -> fill in the form making sure to tick the repo access (first one)
- Make sure to save this token bc you won't get to see it again

##### more git push issues
- I additionally wasn't able to set up git directly from the the jasmin juypterlabs terminal - I had to set up a ssh connection to the jasmin login servers in order to get push to work
- The following links are very helpful to get that set up:
- https://help.jasmin.ac.uk/docs/getting-started/generate-ssh-key-pair/
- https://help.jasmin.ac.uk/docs/getting-started/present-ssh-key/
- https://help.jasmin.ac.uk/docs/interactive-computing/login-servers/#connecting-to-a-sci-server-via-a-login-server

additional advise for windows users:
- using puttygen seemed to work best for me
- If you haven't ssh'ed before you have to enable openssh which if you search optional features in windows search -> add features -> and add openssh client should work
- When SSH'ing into the login server I had to specify the location of my private key explicitly (alternativly you could set up an agent) I did this by adding -I /path/to/key within the ssh -A fred@login-01.jasmin.ac.uk command