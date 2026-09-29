# AWS runbook: train and predict on EC2 (AWS CLI, no SageMaker)

**Why EC2 rather than SageMaker.** The job is a long, multi-stage pipeline. It needs about 300 GB of RAM for the
pandas and GBDT stages and 4 GPUs running 4 cross-encoders in parallel. A single EC2 GPU box driven over SSH is
simpler, cheaper (no SageMaker markup, no idle notebook), and every step can be restarted by hand.

| phase | what runs | instance | wall-clock | approx. cost (on-demand, us-east-1) |
|---|---|---|---|---|
| gbdt | prepare → block → features → GBDT stack → first `matching_results.tsv` | g6e.12xlarge (4× L40S 48 GB, 48 vCPU, 384 GB RAM) | 3–4 h | ≈ $35–45 |
| deep | bi-encoder, re-block, 4 cross-encoders in parallel, stacker | same box | 8–14 h | ≈ $90–150 |

Fallback if you cannot get g6e quota: **g5.12xlarge** (4× A10G 24 GB, 48 vCPU, 192 GB RAM, ≈ $5.7/h). In
`configs/aws.yaml`, set `gbdt_train_s1_frac: 0.25` and use cross-encoder `batch_size: 16` with `grad_accum: 8`.

> Never paste access keys into chat, code or config files. The only place they go is `aws configure` on your
> own machine.

---

## 0. Once: an AWS account with billing enabled
Sign in at https://console.aws.amazon.com. Pick **one region** and use it everywhere. This runbook uses
`us-east-1` (N. Virginia), which has the most GPU capacity.

## 1. Create CLI credentials (about 5 min)
1. Console → **IAM** → *Users* → **Create user**. Name it `ml-challenge`. Do not enable console access.
2. *Permissions* → **Attach policies directly** → tick `AmazonEC2FullAccess`, `AmazonS3FullAccess` and
   `ServiceQuotasFullAccess` → Create user.
3. Open the user → **Security credentials** → **Create access key** → use case **Command Line Interface (CLI)** →
   tick the confirmation → Create → **Download .csv file**. Keep it private.
4. On your Windows laptop (the AWS CLI is already installed at `C:\Program Files\Amazon\AWSCLIV2`), open PowerShell:
   ```powershell
   aws configure
   # AWS Access Key ID:     <from the csv>
   # AWS Secret Access Key: <from the csv>
   # Default region name:   us-east-1
   # Default output format: json
   aws sts get-caller-identity      # must print your account id -> credentials work
   ```

## 2. Request GPU quota. Do this first: approval takes minutes to about a day
New accounts have **0 vCPU** of GPU quota. You need 48 vCPU of "Running On-Demand G and VT instances".
```powershell
# find the quota code (usually L-DB2E81BA)
aws service-quotas list-service-quotas --service-code ec2 --region us-east-1 `
  --query "Quotas[?contains(QuotaName,'On-Demand G and VT')].[QuotaCode,QuotaName,Value]" --output table
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA `
  --desired-value 48 --region us-east-1
# check status later
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region us-east-1 `
  --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" --output table
```
Alternatively, in the Console go to **Service Quotas** → Amazon EC2 → search "Running On-Demand G and VT instances" →
Request increase → 48. In the justification, write "ML model training for a university/industry challenge, one
g6e.12xlarge".

## 3. Key pair and firewall (once)
```powershell
cd $HOME
aws ec2 create-key-pair --key-name ber-key --key-type ed25519 --query KeyMaterial --output text | Out-File -Encoding ascii ber-key.pem
icacls ber-key.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"      # ssh refuses keys readable by others
$MYIP = (Invoke-RestMethod https://checkip.amazonaws.com).Trim()
$SG = aws ec2 create-security-group --group-name ber-ssh --description "ssh for ML challenge" --query GroupId --output text
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr "$MYIP/32"
$SG
```

## 4. Launch the GPU instance
```powershell
# latest Deep Learning Base GPU AMI (NVIDIA driver + CUDA preinstalled), Ubuntu 22.04
$AMI = aws ec2 describe-images --owners amazon `
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" `
  --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text
$AMI
'[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":600,"VolumeType":"gp3","Iops":6000,"Throughput":500,"DeleteOnTermination":true}}]' | Out-File -Encoding ascii disk.json
$IID = aws ec2 run-instances --image-id $AMI --instance-type g6e.12xlarge --key-name ber-key `
  --security-group-ids $SG --block-device-mappings file://disk.json `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ber-train}]" `
  --query "Instances[0].InstanceId" --output text
aws ec2 wait instance-running --instance-ids $IID
$IP = aws ec2 describe-instances --instance-ids $IID --query "Reservations[0].Instances[0].PublicIpAddress" --output text
"$IID  $IP"
```
If you get `InsufficientInstanceCapacity`, retry in a few minutes or add `--placement AvailabilityZone=us-east-1b`
(or c, d, and so on). If you get `VcpuLimitExceeded`, the quota from step 2 is not approved yet.

## 5. Upload code and data (about 1.1 GB)
```powershell
cd "C:\Users\rakes\OneDrive\Desktop\ML\ML Challenge"
powershell -ExecutionPolicy Bypass -File business_entity_resolution\scripts\package_code.ps1   # -> ber_code.zip
scp -i $HOME\ber-key.pem ber_code.zip 6ab10eb3b23ba_student_resource.zip ubuntu@${IP}:~/
ssh -i $HOME\ber-key.pem ubuntu@$IP
```
Then **on the instance**:
```bash
unzip -q ber_code.zip -d ~ && ls ~/business_entity_resolution
bash ~/business_entity_resolution/scripts/aws_setup.sh ~/6ab10eb3b23ba_student_resource.zip   # ~10 min, ends with "SETUP OK"
```

## 6. Run: phase 1, first submission
```bash
tmux new -s ber                      # survives SSH disconnects. Detach: Ctrl-b then d. Re-attach: tmux attach -t ber
cd ~/business_entity_resolution
bash scripts/run_all.sh configs/aws.yaml gbdt
```
Monitor from a second SSH session: `tail -f /data/work/logs/*.log`, `htop`, `nvidia-smi`.
Key numbers to check:
* `/data/work/train/blocking_report.json`: `pair_recall` should be ≥ 0.995 and `avg_candidates` about 20–50.
* `/data/work/train/oof_report_p.json`: `macro_f05` is the validation score, with the `by_country` breakdown.
* The final validator lines print **PASS**.

## 7. Download results to the laptop
```powershell
mkdir results -Force
scp -i $HOME\ber-key.pem "ubuntu@${IP}:/data/work/output/*.tsv" results\
scp -i $HOME\ber-key.pem "ubuntu@${IP}:/data/work/train/*.json" results\
```
Upload `results\matching_results.tsv` to the challenge portal.

## 8. Phase 2: deep model zoo (optional, for the last points)
```bash
tmux attach -t ber
bash scripts/run_all.sh configs/aws.yaml deep
```
Compare `/data/work/train/stack_report.json` (`macro_f05` against `gbdt_only_same_holdout`) before submitting the
new file. The previous GBDT file is kept as `matching_results.gbdt.tsv`.

## 9. STOP PAYING when done
```powershell
# first copy anything you need: /data (instance store) is wiped on stop
aws ec2 terminate-instances --instance-ids $IID
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].[InstanceId,InstanceType]" --output table
```
A stopped (not terminated) instance still bills for its 600 GB EBS volume (about $50 a month). Terminate when done.

## Troubleshooting
| symptom | fix |
|---|---|
| `ssh: Permissions for 'ber-key.pem' are too open` | re-run the `icacls` line from step 3 |
| SSH timeout | your IP changed; re-run the `$MYIP` and `authorize-security-group-ingress` lines |
| `CUDA out of memory` in a cross-encoder | halve `batch_size` and double `grad_accum` for that model in `configs/aws.yaml` |
| process `Killed` (RAM) | lower `gbdt_train_s1_frac` (for example 0.25) or `blocking.max_cand_per_s1` (for example 40) |
| want a cheaper box for phase 1 only | r7i.12xlarge (CPU, 384 GB) handles gbdt; deep models need a GPU box |
