# Round 3 teacher data: QA pairs and summaries written by Claude (Fable 5)
# after reading the EF Mandate, i.e. distillation from a frontier model into
# Qwen2.5-1.5B. Same jsonl schema as ttt_augment_3080.py so ttt_train2_3080.py
# consumes it unchanged. Run: py -3.10 ttt_synth_fable.py > ttt_synth_fable.jsonl
import json

QA = [
    # I. Ethereum
    ("What was Ethereum born out of, according to the EF Mandate?", "A dream for freedom, for all who are ready to grasp it with their own hands."),
    ("Which two vital tools did Ethereum's creators think the armamentarium of freedom was missing?", "Self-sovereign computation, and the computational ability to coordinate at scale without violating anyone else's self-sovereignty."),
    ("Per the EF Mandate, what was the first application of Ethereum?", "Money, because money requires both coordination and self-sovereignty."),
    ("What does 'The World Computer' mean in the EF Mandate?", "Ethereum as humanity's common computational substrate that anyone can interact with trustlessly, permissionlessly, and persistently."),
    ("What is Ethereum's second promise per the Mandate?", "Allowing infrastructures of self-sovereign coordination to arise and thrive in any form imaginable, without violating any individual's freedom."),
    ("Why does the Ethereum Foundation exist, in one sentence from chapter I?", "To ensure Ethereum remains resilient enough to be a liberatory technology."),
    # II. Our Role
    ("What is the Ethereum Foundation's role relative to the Ethereum project?", "It is the original steward of the Ethereum project, not its parent, owner, or ruler."),
    ("What metaphor does the EF use for what Ethereum grew into?", "An infinite garden that countless participants use to grow their own projects."),
    ("What is the walkaway test in the EF Mandate?", "Ethereum's protocol and core application layers should be robust and trustless enough to keep functioning and evolving even if the Foundation and today's core developers disappeared tomorrow."),
    ("Is the Ethereum Foundation a for-profit organization?", "No, it describes itself as a real non-profit, independent, with no other agenda, rejecting temptations around flows of value."),
    ("What does the EF call its enduring assets?", "Its legitimacy and virtue, which it will not risk or squander."),
    ("What is the EF's bottom line according to the Mandate?", "Not profit, growth, or blind adoption: the bottom line is the mission of securing Ethereum's resilience."),
    ("What are the EF's primary and secondary measures of success?", "How much self-sovereignty, and how much sovereignty-preserving coordination at scale, Ethereum resiliently enables, both with and without the Foundation."),
    ("Who is the EF Mandate primarily written for?", "Members of the Ethereum Foundation, as a clarification of purpose and a practical guide for translating mission and principles into action."),
    # III. Our Mandate
    ("What is the first aim of the EF's twofold mandate?", "To ensure Ethereum becomes and stays a decentralized and resilient tool for self-sovereignty, where a user has final say over their identities, assets, actions, and agents."),
    ("What is the second aim of the EF's twofold mandate?", "Scaling the guaranteed availability of self-sovereignty to users ready to exercise it directly, without violating anyone else's."),
    ("List the four CROPS properties named in the EF Mandate.", "Censorship Resistance; Open Source and Free, as in Freedom; Privacy; Security."),
    ("How must the CROPS properties be treated per the Mandate?", "As an indivisible whole, the sine qua non of all Ethereum's development priorities, which cannot be displaced."),
    ("For what time horizon is the EF Mandate written?", "A thousand-year horizon; it starts as high as possible to slow long-run erosion of standards over centuries."),
    ("What does the Mandate say about standards over time?", "Like water, standards flow from high to low and are far easier to lose than to regain, so the EF starts as high as it can."),
    ("Does the EF expect compromises to its Mandate within our lifetimes?", "No, it does not expect any material compromise within our lifetimes."),
    ("What does the EF believe about self-sovereignty as a strategy?", "That it is positive, positive-sum, and that self-sovereignty at scale is the dominant positive-sum strategy."),
    # IV. Principles for Action - technical pillar
    ("How many pillars and principles support the EF Mandate?", "Two pillars, a technical pillar and a social pillar, each comprising four principles."),
    ("Define censorship resistance as the EF Mandate does.", "No actor can selectively exclude valid use or break functionality, including by gaining durable, non-competitive control of critical mechanisms."),
    ("How should all EF work be architected per the censorship-resistance principle?", "To be maximally unstoppable and to function without centralized intermediaries or kill switches."),
    ("What does 'Open Source and Free, as in Freedom' require?", "No privileged code or hidden specifications; all work public, auditable, and forkable."),
    ("What licensing pledge must EF-supported projects make?", "That they will not change their open source or copyleft license in the future; permissive licenses accepted, viral copyleft appreciated, source-available not tolerated."),
    ("How does the EF Mandate define the privacy principle?", "User data is not exposed beyond necessity or against their interests; maximal privacy should become the default."),
    ("What does the Mandate say privacy is really about?", "Not total concealment, but freedom and true consent: choosing what to disclose to whom, on one's own terms, on a base of freely available unconditional privacy."),
    ("How does the EF Mandate define the security principle?", "Things must do what they claim to do, no more and no less."),
    ("Why does security require simplicity per the Mandate?", "A protocol is not trustless if only a few people can understand why it is secure; work must be verifiable to many, so lines of code and dependencies are responsibly minimized."),
    ("What does 'governance minimization' mean in the security principle?", "No social layer should override protocol guarantees lightly."),
    # IV. Social pillar
    ("Name the four principles of the EF Mandate's social pillar.", "Principled Alignment, Discipline, Right Association, and Big Picture."),
    ("What does the EF Mandate say about a billion users in a centralized silo?", "It is not a success but a failure of mission, as is enshrining centralized extraction pipelines in the protocol."),
    ("What does the Discipline principle demand?", "Truth-seeking and beauty-seeking work, technical rigor, excellence, creativity, and courage to make unpopular principled decisions."),
    ("How does the EF admit mistakes per the Discipline principle?", "With humility, grace, and an honest, clear explanation of why views changed and what the new views are."),
    ("What is the Right Association principle?", "Who the EF works with is itself a principled choice: it prioritizes people who share its principles, spread them, and document openly."),
    ("Per Right Association, which projects does the EF prefer to work closely with?", "Those that also actively work to achieve independence from the Foundation."),
    ("What does the Big Picture principle say about Ethereum's horizon?", "It is broader than crypto: Ethereum's promise only holds if it serves self-sovereignty beyond any one subculture, asset class, or industry."),
    ("Who does the Mandate name as natural allies of the World Computer?", "Open source projects, privacy and cryptography researchers, civil liberties defenders, educators, public-interest technologists, builders of resilient local communities, and quiet maintainers of civilization."),
    # V. Carrying out the work
    ("Summarize the EF's operating approach in one phrase from the Mandate.", "A process of subtraction for resilience: bias toward work that makes the Foundation less necessary over time."),
    ("What is the Only-EF Rule?", "The EF focuses on critical tasks that have no other natural home and that no other ecosystem actor can or will reliably undertake."),
    ("Give examples the Mandate lists under the Only-EF Rule.", "Core protocol upgrades, long-horizon research, neutral multi-client specs and tests, public-good security work, crisis coordination, preventing chokepoints, and core dev tooling with no sustainable owner."),
    ("What is Handoff for Ecosystem Maturity?", "As soon as a function can be successfully managed by an aligned community actor, the EF facilitates that transition so capability diffuses instead of concentrating."),
    ("What does Compounding Effects mean in the EF's approach?", "Prioritizing upstream, high-leverage work whose research, documentation, coordination, and infrastructure can be freely reused, extended, and operated independently."),
    ("What is Subtraction as Success?", "The EF's goal is to reduce the Foundation's relative influence over time, ensuring Ethereum grows decentralized and robust enough to outgrow and outlast it."),
    ("What memorable line captures subtraction in the Mandate?", "The more Ethereum succeeds, the tinier we become; if Ethereum fails, so too will we perish. Subtraction will occur either way, so we choose success."),
    ("What single sentence summarizes the EF's limits?", "We do for Ethereum what Ethereum is meant to do for its users."),
    ("List at least five of the nine 'We are NOT' items in the EF Mandate.", "Not a Corporate, not a Kingmaker, not an Accreditation Body, not a Product Studio, not a Marketing Agency, not the Boss, not a Government or Regulatory Body, not a Casino, not Opportunists."),
    ("Why does the Mandate say the EF is not a casino?", "It does not encourage people to take life-wrecking risk through personal-debt hyper-gambling; Ethereum should be a foundation for a secure and free life, and debt promotes the opposite."),
    ("What does 'We are NOT the Boss' mean?", "The EF cannot force hard forks or protocol changes; it is opinionated only to advocate and propose what is best for the mission."),
    ("Through which partially centralized surfaces does most Ethereum use flow today, per the Mandate?", "Wallets, RPC providers, relays to the MEV-industrial complex, app stores, exchanges, institutions, and the social defaults around them."),
    ("What two approaches to growing CROPS adoption does the Mandate contrast?", "An incrementalist approach that shows CROPS increases value at scale, and a nativist (CROPS-native) approach that directly grows and distributes CROPS; the CROPS-native path is the default priority."),
    ("When does the EF leave space for the incrementalist approach?", "Only in tightly bounded circumstances: as a tactical intervention that durably reduces central control, avoids deeper entrenchment, and accelerates a credible fully-principled alternative."),
    ("Complete the aphorism: 'Adoption can be earned over time, but...'", "principled ground once ceded is far harder to regain."),
    ("Which ancient school does the Mandate cite for openly distributing defense, and how?", "The Mohists, who authored and widely distributed manuals that helped all cities defend themselves, shifting the balance from offense to defense to reduce suffering."),
    ("How does the EF's approach differ from the Mohists'?", "The Mohists directly intervened in conflicts based on their own judgment; the EF's approach is closer to writing the manuals and making them available without intervening in individual conflicts."),
    ("What is de-totalization in the EF Mandate?", "Building toward a world in which no organization, system, or moral order has total dominance over any individual life; the Mandate calls it the most reliably good aim."),
    ("What guiding question does the Mandate pose for tradeoffs?", "Does this make Ethereum and its users less susceptible to capture over time, or does it normalize capture in exchange for reach?"),
    # VI. Resolving quandaries
    ("How many timeless tensions does the Resolving Quandaries chapter illustrate?", "Five."),
    ("Quandary 1: when two technically credible paths compete, which wins?", "The one that removes points of leverage, not the one that can be shipped faster."),
    ("What lesson closes quandary 1?", "It is not sufficient that a solution works today; it also needs to not become a chokepoint tomorrow."),
    ("Quandary 2 asks designers to think about what?", "Higher-order effects across layers, so capture points are not simply displaced beyond the narrow focus or turned into externalities."),
    ("Which five cross-layer scenarios does quandary 2 walk through?", "Scale, account types, native privacy support at the protocol layer, transaction protections at the protocol layer, and aggregation of cryptographic objects."),
    ("Why does the Mandate favor native protocol privacy over layered constructions?", "Protocol-native privacy greatly increases the anonymity set; no construction layered on top could match it."),
    ("Why is aggregation of cryptographic objects a centralization risk?", "High fixed costs make the aggregation market likely monopolistic, a centralized chokepoint, which protocol-level batched aggregation would remove."),
    ("What higher bar does the Mandate set for protocol changes?", "Protocol improvements that bear any risk at all to CROPS properties are held to a much higher bar and evaluated with greater caution."),
    ("Quandary 3: what is the default in adversarial user environments?", "Empowering user agency, not solutions that weaken it; no high priests installing restrictions users never opted into or can't opt out of."),
    ("What bad wallet example does quandary 3 give?", "A default-on 'safe mode' with dark patterns: silently blocking contracts, steering users to preferred venues, unmodifiable preinstalled whitelists, or an AI copilot using an uninspectable proprietary model that reports user actions home silently."),
    ("What user-controlled defenses does CROPS push instead?", "Independent locally-verifiable filters with transparent rules, multiple community-created whitelists and blacklists with clear override paths, and private-by-default tool use including AI components."),
    ("What line summarizes quandary 3's goal?", "The goal is not to sanitize the environment; it is to keep users sovereign inside it."),
    ("Quandary 4: what is the 'zero option'?", "For every affordance with an intermediated path, any possible intermediary-free path must be built and must remain credible and accessible; the EF does not skip this step."),
    ("What is the north star of quandary 4?", "Disintermediation: eliminate intermediaries where possible, and where unavoidable keep intermediary roles open, plural, bounded, and verifiable."),
    ("In the identity example, what narrower alternative to full identity does the Mandate suggest?", "If only sybil resistance is needed, users could give a zero-knowledge proof of ETH ownership or post a zero-knowledge security deposit instead of revealing identity."),
    ("Quandary 5: how does the EF judge which teams to back?", "By looking past short-term output and social cues to patterns of choices and revealed preferences, not anchoring on polish, credentials, or social proof."),
    ("What red flags does quandary 5 list despite CROPS language?", "Closed components, whitelists, soft-default routing, discretionary upgrade ability, and dependency-heavy integrations."),
    ("Does the EF require first versions to be complete?", "No; they only need to stay live, since open source building lets subsequent teams improve or finish a strong design path."),
    # VII. The Future & VIII. Closing
    ("What two bad choices does the Future chapter say people have been pushed to accept?", "Accepting rule from the top by macro-sovereigns who already hold power, or responding without a principled aim by burning it down, retreating, or defecting opportunistically."),
    ("What does Ethereum reject, per the Future chapter?", "The idea that there is no alternative."),
    ("What are the sanctuaries described in the Future chapter?", "Resilient shelters from the contest of power that individuals, families, and communities independently build with Ethereum, where anyone can live neither oppressed nor oppressing."),
    ("Who are the EF's closest collaborators per the Future chapter?", "Those working directly on privacy, verifiability, and programmable cryptography, with open silicon and alternative networks in the middle distance."),
    ("What closing slogan does the Future chapter end with?", "Alternatives exist. Trust hope, embrace resilience."),
    ("What is the EF here to do per the Closing chapter?", "To uncapture the individual and entrench their freedoms of association, not to capture markets, corporates, or states."),
    ("Summarize the 'rug' passage of the Closing chapter.", "Ethereum exists so nothing can rug you: other people, society, governments, corporations, institutions, AI, mountain men, even your family, and so you don't accidentally rug yourself."),
    ("What has the EF been entrusted with, per the Closing?", "The torch of liberty, to keep burning bright until the time comes to pass it on as it was passed to them."),
    ("What is the EF building per the Closing chapter's famous line?", "Nothing less than the machinery of freedom, not just for today but for the next thousand years."),
    ("Which Italian sentence closes the EF Mandate, and where is it from?", "E quindi uscimmo a riveder le stelle, the closing line of Dante's Inferno: and so we emerged to see the stars again."),
    ("What does the Mandate say about the road to the stars?", "The work will sometimes be thankless, arduous, and lonely, but every road to the stars first passes through darkness."),
]

TEXTS = [
    "The EF Mandate (2026) is the Ethereum Foundation's constitution-style document. Structure: I. Ethereum, II. Our Role, III. Our Mandate, IV. Principles for Action, V. Carrying Out the Work, VI. Resolving Quandaries, VII. The Future, VIII. Closing. Its core claims: Ethereum is a tool for self-sovereignty; the Foundation is steward, not ruler; four CROPS properties (Censorship Resistance, Open Source, Privacy, Security) are non-negotiable; the EF succeeds by subtracting itself.",
    "The mandate is twofold. First: keep Ethereum a decentralized, resilient tool for self-sovereignty, where users have final say over identities, assets, actions, and agents, guaranteed by the indivisible CROPS properties. Second: scale unstoppable self-sovereignty to everyone who chooses it, at the scale and in the form they want, without violating anyone else's. It is written for a thousand-year horizon, starting as high as possible because standards flow from high to low.",
    "The EF's operating approach is subtraction for resilience, guided by five ideas: the Only-EF Rule (do only critical work with no other natural home), Handoff for Ecosystem Maturity, Independent Inspiration and Reliability (mission as glue, not hierarchy), Compounding Effects (upstream, reusable work), and Subtraction as Success (the more Ethereum succeeds, the tinier the EF becomes).",
    "The EF's limits are stated as nine negations: not a Corporate, not a Kingmaker, not an Accreditation Body, not a Product Studio, not a Marketing Agency, not the Boss, not a Government or Regulatory Body, not a Casino, not Opportunists. In short: we do for Ethereum what Ethereum is meant to do for its users.",
    "The five quandary rules: 1) pick the path that removes leverage, not the one that ships faster; 2) judge proposals by cross-layer, higher-order effects so capture isn't displaced; 3) in adversarial environments default to empowering user agency, never paternalistic restrictions; 4) guarantee the zero option, a credible intermediary-free path for every intermediated affordance, with disintermediation as north star; 5) back teams by revealed preferences, not polish or social proof.",
    "Two memorable cultural references anchor the Mandate: the Mohists, who wrote and distributed city-defense manuals to shift the offense-defense balance and reduce suffering (the EF writes the manuals but does not intervene in individual conflicts), and Dante, whose line 'E quindi uscimmo a riveder le stelle' closes the document: every road to the stars first passes through darkness.",
]

for q, a in QA:
    print(json.dumps({"type": "qa", "q": q, "a": a}, ensure_ascii=False))
for t in TEXTS:
    print(json.dumps({"type": "text", "text": t}, ensure_ascii=False))
