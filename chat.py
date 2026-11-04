#Enterprise RAG - Console chat
import config
from rich import print
from rag_chain import ask

# 🔐Role menu
role_map = {
    "1" : {"username" : "dev_employee", "role" : "employee", "department" : "engineering"},
    "2" : {"username" : "hr_manager", "role" : "manager", "department" : "hr"},
    "3" : {"username" : "admin", "role" : "admin", "department" : "all"},
}

print("[bold cyan]🏢 Enterprise RAG Intelligence Platform[/bold cyan]")
print("1. Employee (engineering)")
print("2. Manager (hr)")
print("3. Admin")
choice = input("Choose your role (1/2/3): ").strip()
user = role_map.get(choice, role_map["1"])
print(f"Logged in as {user['username']} ({user['role']})\nType 'exit' to quit.")

while True:
    question = input("\nYou: ").strip()
    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        break

    result = ask(question, user)
    print(f"\n[bold green]Assistant:[/bold green] {result['answer']}")
    if result["sources"]:
        print("\n📚 Sources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['source']} (page {s['page']}, {s['department']}/{s['access_level']})")
