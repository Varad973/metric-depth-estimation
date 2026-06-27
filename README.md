# ☁️ SaaS Cloud with HDFS-style Storage + AES Encryption

## 🎯 What This Does
- **cloud_server.py** = Your cloud (the "SaaS" — runs on one machine)
- **cloud_client.py** = User's program (uploads/downloads files)
- Files are split into **blocks** (like HDFS does)
- Each block is **encrypted with AES** before uploading
- Blocks are decrypted and rejoined when downloaded

## 📦 Setup (One-Time)

### Step 1: Install Python libraries
```bash
pip install flask requests cryptography
```

That's it. No Hadoop install, no AWS needed for testing.

---

## 🚀 How to Run

### Option A — Both files on SAME machine (testing)

**Terminal 1:** Start the cloud server
```bash
python cloud_server.py
```
You'll see: `Listening on : http://localhost:5000`

**Terminal 2:** Run the client
```bash
python cloud_client.py
```
Use the menu: 1 to upload, 2 to download, 3 to list.

### Option B — Cloud on AWS EC2, Client on your laptop

1. Launch an Ubuntu EC2 instance (free tier is fine)
2. In the EC2 Security Group → allow inbound port 5000
3. SSH into EC2:
   ```bash
   sudo apt update && sudo apt install python3-pip -y
   pip3 install flask cryptography
   # copy cloud_server.py to EC2
   python3 cloud_server.py
   ```
4. On your laptop, edit `cloud_client.py`:
   ```python
   SERVER_URL = "http://<EC2-PUBLIC-IP>:5000"
   ```
5. Run the client locally.

### Option C — Over LAN (your lab requirement)

1. Run `cloud_server.py` on **one lab PC** (find its IP with `ipconfig` / `ifconfig`)
2. On other PCs, edit `cloud_client.py`:
   ```python
   SERVER_URL = "http://<server-pc-ip>:5000"
   ```
3. Make sure firewall allows port 5000.

---

## 🧪 Demo Steps (For Your Practical Exam)

1. Start the server in one terminal
2. Create a test file:
   ```bash
   echo "Hello cloud, this is my secret data!" > test.txt
   ```
3. Run the client → choose `1` → enter `test.txt`
4. Open `cloud_storage/` folder → you'll see encrypted blocks (open one in Notepad — it's gibberish! ✅ proves encryption works)
5. Run the client → choose `2` → enter `test.txt`
6. Check `downloads/test.txt` — original content restored! ✅

---

## 💡 What to Say in Viva

| Question | Answer |
|----------|--------|
| What is SaaS here? | The cloud server provides "file storage as a service" over HTTP |
| How is HDFS implemented? | Files are split into fixed-size blocks (64KB) and stored separately, with metadata tracking block→file mapping (like NameNode) |
| What encryption is used? | AES (via Fernet from `cryptography` library) — symmetric encryption |
| Why encrypt before upload? | So even if cloud storage is compromised, attackers can't read files without the key |
| How is data split? | File is read in 64KB chunks; each chunk becomes one block |
| Why a metadata.json file? | It mimics HDFS NameNode — keeps track of which blocks belong to which file |
| Cloud Controller? | The Flask server with routes for upload/download/list — it controls all cloud operations |

---

## 📁 Block Size Note
Real HDFS uses **128 MB blocks**. We use **64 KB** so you can test with small files.
To change: edit `BLOCK_SIZE` in `cloud_client.py`.
