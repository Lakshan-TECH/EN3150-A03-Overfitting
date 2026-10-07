# Git guide for group "Overfitting"

## One shared repository (not four)
Create ONE repository. Every member is a collaborator and commits from their OWN GitHub account, so the history
shows each person's work. The lecturer only needs this one link plus each member's profile link (already in the report).

## 1. One member creates the repository (e.g. Abishek)
1. GitHub -> New repository -> name: `EN3150-A03-Overfitting` -> Public (or Private and invite the lecturer) ->
   do NOT tick "Add a README" (this folder already has one).
2. Settings -> Collaborators -> Add people: Lakshan-TECH, sampavany, Santhosh-04-S. Each must accept the invitation.
3. In this folder:
```bash
git init
git add .
git commit -m "Add training script, requirements and README"
git branch -M main
git remote add origin https://github.com/<owner>/EN3150-A03-Overfitting.git
git push -u origin main
```

## 2. Every other member
```bash
git clone https://github.com/<owner>/EN3150-A03-Overfitting.git
cd EN3150-A03-Overfitting
git config user.name  "Your Name"
git config user.email "the-email-linked-to-YOUR-GitHub-account"   # otherwise commits do not appear on your profile
```

## 3. Daily workflow (each member, for their own part)
```bash
git pull                      # get the others' work first
# ... do your work ...
git add <files you changed>
git commit -m "Short, specific message"
git push
```
Commit when you actually finish something (a section, a figure, a fix). Do not backdate commits or push one big
commit at the end: the assignment checks that work was spread over a reasonable period.

## 4. Never commit
`data/`, `outputs/*.pt`, `.venv/` (already in .gitignore).

## 5. Suggested things for each member to commit (match what you really do)
* Abishek: data preparation notes, dataset figures, optimizer-study notes
* Luckshan: architecture notes, parameter/MAC verification, limitations section
* Sampavi: training/evaluation results, loss curves, confusion-matrix analysis
* Santhosh: SOTA fine-tuning results, final comparison, report PDF in docs/
