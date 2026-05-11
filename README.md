# 🐞 BugSearchApp — Semantic Bug Search Desktop Application

BugSearchApp is a Windows desktop application that enables **semantic search over Azure DevOps bugs** using local embeddings.

Instead of keyword matching, it understands the **meaning of error descriptions** and returns the most relevant bugs instantly.

---

## 🚀 Features

- 🔍 Semantic bug search using SentenceTransformers (`all-MiniLM-L6-v2`)
- ⚡ Fast local search (offline, no API calls during search)
- 🔄 Incremental refresh from Azure DevOps (only new/updated bugs)
- 🧠 Vector embeddings stored locally (`.npy`)
- 📊 Metadata stored in CSV for fast lookup
- 🖥️ Tkinter-based desktop UI
- 🔗 Clickable Azure DevOps links
- 🧪 Built-in Diagnostics tool
- 📦 Packaged as a Windows installer (no Python required)

---

## 🏗️ Tech Stack

- Python 3.11
- Tkinter (UI)
- SentenceTransformers (embeddings)
- NumPy (vector operations)
- Pandas (data handling)
- Azure DevOps REST API (WIQL)
- PyInstaller (EXE build)

# 🐞 BugSearchApp — Semantic Bug Search Desktop Application

BugSearchApp is a Windows desktop application that enables **semantic search over Azure DevOps bugs** using local embeddings.

Instead of keyword matching, it understands the **meaning of error descriptions** and returns the most relevant bugs instantly.

---

## 🚀 Features

- 🔍 Semantic bug search using SentenceTransformers (`all-MiniLM-L6-v2`)
- ⚡ Fast local search (offline, no API calls during search)
- 🔄 Incremental refresh from Azure DevOps (only new/updated bugs)
- 🧠 Vector embeddings stored locally (`.npy`)
- 📊 Metadata stored in CSV for fast lookup
- 🖥️ Tkinter-based desktop UI
- 🔗 Clickable Azure DevOps links
- 🧪 Built-in Diagnostics tool
- 📦 Packaged as a Windows installer (no Python required)

---

## 🏗️ Tech Stack

- Python 3.11
- Tkinter (UI)
- SentenceTransformers (embeddings)
- NumPy (vector operations)
- Pandas (data handling)
- Azure DevOps REST API (WIQL)
- PyInstaller (EXE build)
- Inno Setup (installer packaging)

## ⚙️ How It Works

### 🔍 Search Flow

1. User enters an error / description
2. Input is converted into embedding
3. Compared with stored embeddings (vector similarity)
4. Top matching bugs are returned
5. User can open bug directly in Azure DevOps

<img width="1251" height="784" alt="image" src="https://github.com/user-attachments/assets/71f28f1f-c822-4a51-8135-bb32b8b029a3" />


---

### 🔄 Refresh Flow

1. Runs WIQL query:

System.ChangedDate > last_refresh_time

2. Fetches only new/updated bugs
3. Builds semantic text
4. Generates embeddings
5. Appends to existing index
6. Updates metadata + fingerprints
7. Reloads search index

---

## 🧠 Data Storage

| File | Purpose |
|------|--------|
| `bug_embeddings.npy` | Vector embeddings |
| `bug_metadata.csv` | Bug details |
| `bug_fingerprints.json` | Change detection |
| `refresh_state.json` | Last refresh timestamp |

---
- Inno Setup (installer packaging)

---
