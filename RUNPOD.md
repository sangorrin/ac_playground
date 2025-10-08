# Prepare RUNPOD

## [ONCE] Generate a new key pair (ed25519)

Create the private and public key, and upload the public key to runpod.
```bash
ssh-keygen -t ed25519 -a 100 -C "runpod" -f ~/.ssh/runpod_ed25519
chmod 600 ~/.ssh/runpod_ed25519

# Copy the public key and paste it in RunPod Settings → SSH Public Keys → Update
cat ~/.ssh/runpod_ed25519.pub
```

## Choose a runpod

- clikc Pods > Deploy a Pod
  - Choose: GPU, Secure Cloud, Any Region
    [Note] Do not choose Community Cloud (network problems)
  - Additional Filters: 
    - select same CUDA version as template (e.g. 12.8)
  - Select RTX 4090
  - Set a pod name: constant_freevc_daniel
  - Select Pod Template
    - Runpod Pytorch 2.8.0
    - runpod/pytorch:1.0.1-cu1281-torch280-ubuntu2404
  - Select On-Demand
  - Check SSH Terminal Access and Jupyter

## [Optional] Create an ssh config for easy access to pods

Start the pod and get the SSH IP/PORT from “Connect” tab.
Then fill the ssh config so you can access by SSH easily.
[Note] If you stop/restart the pod, they may change.
[Alt] Use the non-IP ssh command in the connect tab.

```bash
cat >> ~/.ssh/config <<'EOF'
Host runpod-1
  HostName <POD_IP>
  Port <PORT>
  User root
  IdentityFile ~/.ssh/runpod_ed25519
  IdentitiesOnly yes
EOF

ssh runpod-1
```

## [Once] Runpod API key

Create a Runpod API in the UI with Read/Write permissions.

[Optional] Set restricted permissions:
- api.runpod.io/graphql: None
- api.runpod.ai: Restricted
    Pods → Read (list/get status)
    Pods → Write → Start / Stop
    (Do not grant Create/Delete/Terminate unless you need them.)
    (Optional) Pods → Resume if you ever resume a stopped pod.
    Templates → Read (to launch from a template)


Install RunPod CLI and set API key:
```bash
export RUNPOD_API_KEY=YOUR_API_KEY # store securely
brew install runpod/tap/runpodctl
runpodctl config --apiKey $RUNPOD_API_KEY
```

Usage:
```bash
runpodctl get pod # check your POD_ID
export POD_ID=YOUR_PODID
runpodctl start pod $POD_ID
runpodctl stop pod  $POD_ID
```
