from tasks.pull_solicitations_task import PullSolicitationsTask
from tasks.preprocess_task import PreprocessTask
from tasks.archive_solicitations_task import ArchiveSolicitationsTask
from rag.faiss_store import FaissStore

class SolicitationAgent:
    def __init__(self, config, store: FaissStore, *, dry_run: bool = False):
        self.api_key = config["SAM_API_KEY"]
        self.pull_task = PullSolicitationsTask(self.api_key)
        self.archive_task = ArchiveSolicitationsTask(
            config.get("MINIO_ACCESS_KEY"),
            config.get("MINIO_SECRET_KEY"),
            dry_run=dry_run,
        )
        self.preprocess_task = PreprocessTask()
        self.store = store

    def run(self):
        print("🔍 Pulling solicitations...")
        opportunities = self.pull_task.execute()
        print(f"✅ Pulled {len(opportunities)} solicitations.")

        if not opportunities:
            print("⚠️ No solicitations found. Exiting early.")
            return

        print("📦 Archiving raw JSON responses...")
        self.archive_task.execute(opportunities)

        print("🧹 Preprocessing documents...")
        processed_docs = self.preprocess_task.execute(opportunities)
        print(f"✅ Preprocessed {len(processed_docs)} documents.")

        if not processed_docs:
            print("⚠️ No processed documents to embed. Exiting early.")
            return

        print("🧠 Embedding and storing in FAISS...")
        self.store.overwrite_documents(processed_docs)
        print("✅ Stored active solicitations in FAISS.")
