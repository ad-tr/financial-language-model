import wikipedia
from tqdm import tqdm

wikipedia.set_lang("fr") 
data = []

hubs = [
    "CAC 40", "SBF 120", "Dow Jones Industrial Average", "NASDAQ-100", 
    "Liste des cryptomonnaies", "Histoire des bourses de valeurs",
    "Crise financière", "Vocabulaire boursier"
]

seeds = [
    "Bourse (économie)", "Action (finance)", "Obligation (finance)", "Dividende",
    "Capitalisation boursière", "Analyse technique", "Analyse fondamentale",
    "Chandelier japonais", "Indicateur technique", "Moyenne mobile", "RSI (finance)",
    "Effet de levier", "Vente à découvert", "Dark pool", "Haute fréquence (trading)",
    "Gestion d'actifs", "Fonds indiciel (ETF)", "Hedge fund",
    
    "Inflation", "Déflation", "Taux d'intérêt", "Politique monétaire",
    "Banque centrale", "Réserve fédérale des États-Unis", "Banque centrale européenne",
    "Christine Lagarde", "Jerome Powell", "Assouplissement quantitatif",
    "Produit intérieur brut", "Récession économique", "Dette publique",
    
    "Apple", "Microsoft", "Alphabet (entreprise)", "Amazon (entreprise)", 
    "NVIDIA", "Tesla (entreprise)", "Meta Platforms",
    "Berkshire Hathaway", "BlackRock", "JPMorgan Chase", "Goldman Sachs",
    "LVMH", "TotalEnergies", "Sanofi", "BNP Paribas", "Société générale",
    
    "Cryptomonnaie", "Bitcoin", "Satoshi Nakamoto", "Halving (Bitcoin)",
    "Ethereum", "Vitalik Buterin", "Blockchain", "Contrat intelligent",
    "Finance décentralisée", "Binance", "Coinbase", "FTX (entreprise)",
    "Solana (blockchain)", "Ripple (entreprise)", "Cardano (cryptomonnaie)",
    "Stablecoin", "Tether (cryptomonnaie)", "USDC (cryptomonnaie)",
    "Non-fungible token", "Web3", "Metavers", "Minage (cryptomonnaie)",
    "Preuve d'enjeu", "Preuve de travail", "Monnaie numérique de banque centrale",
    
    "Crise économique", "Krach de 1929", "Bulle Internet", "Crise des subprimes",
    "Crise de la dette dans la zone euro", "Faillite de Lehman Brothers",
    "Scandale Enron", "Système de Ponzi", "Bernard Madoff", "Tulipomanie",
    "Loup de Wall Street", "Warren Buffett", "George Soros"
]

wiki_pages = seeds.copy()
seeds_set = set(seeds)
chars = 0

for hub in hubs:
    suggest = wikipedia.search(hub)[0]
    links = wikipedia.page(suggest, auto_suggest=False).links
    for link in links:
        if link not in seeds_set:
            wiki_pages.append(link)
            seeds_set.add(link)
            print(f'{len(wiki_pages)}. Ajout de "{link}"')

for seed in seeds:
    suggests = wikipedia.search(seed)
    for suggest in suggests[:3]:
        if suggest not in seeds_set:
            wiki_pages.append(suggest)
            seeds_set.add(suggest)
            print(f'{len(wiki_pages)}. Ajout de "{suggest}"')

print("---- Récupération des pages ----")

loop = tqdm(wiki_pages, desc="Moissonnage de données", unit="page", colour="green")
for wiki_page in loop:
    loop.set_description(f"Traitement : {wiki_page[:25]}...")
    content = ""
    title = ""
    try:
        try:
            page = wikipedia.page(wiki_page, auto_suggest=True)
            content = page.content
            title = page.title
        except:
            page = wikipedia.page(wiki_page, auto_suggest=False)
            content = page.content
            title = page.title
    except:
        continue
    
    chars = chars + len(content)
    print(chars)
    
    if content != "":
        with open(f"./data/corpus/corpus.txt", "a", encoding="utf-8") as file:
            file.write("</BOS>")
            file.writelines(content)
            file.write("</EOS>")
