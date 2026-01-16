"""
Analyze mistake patterns between OLMo-2-0425-1B and Qwen2.5-0.5B models
on college_medicine subject (Questions 109-173)
"""

# Question-by-question results for college_medicine (Q109-173)
# Format: question_num: (olmo_correct, qwen_correct, olmo_answer, qwen_answer, correct_answer, topic_category)

results = {
    # Q109 - not in detailed output for OLMo
    110: (False, False, 'A', 'A', 'D', 'physiology'),  # Blood lactate - BOTH WRONG, SAME ANSWER
    111: (False, False, 'A', 'A', 'D', 'physiology'),  # Type I muscle fibres - BOTH WRONG, SAME ANSWER
    112: (False, False, 'C', 'C', 'A', 'physics'),     # Gas volume - BOTH WRONG, SAME ANSWER
    113: (True, True, 'C', 'C', 'C', 'physics'),       # Bernoulli's principle - BOTH CORRECT
    114: (False, True, 'C', 'A', 'A', 'physiology'),   # Sodium bicarbonate - OLMo wrong, Qwen correct
    115: (False, False, 'B', 'B', 'D', 'psychology'),  # Gender/orientation - BOTH WRONG, SAME ANSWER
    116: (False, True, 'B', 'C', 'C', 'biochemistry'), # Intracellular buffer - OLMo wrong, Qwen correct
    117: (False, False, 'C', 'C', 'B', 'psychology'),  # Signal Detection - BOTH WRONG, SAME ANSWER
    118: (False, True, 'C', 'A', 'A', 'sociology'),    # Subculture - OLMo wrong, Qwen correct
    119: (False, False, 'C', 'C', 'D', 'biochemistry'),# Prosthetic groups - BOTH WRONG, SAME ANSWER
    120: (False, True, 'C', 'A', 'A', 'genetics'),     # Codons - OLMo wrong, Qwen correct
    121: (False, False, 'C', 'C', 'B', 'physiology'),  # Blood lactate soccer - BOTH WRONG, SAME ANSWER
    122: (True, True, 'C', 'C', 'C', 'neuroscience'),  # Neurons - BOTH CORRECT
    123: (True, False, 'B', 'A', 'B', 'nutrition'),    # Creatine supplements - OLMo correct, Qwen wrong
    124: (False, True, 'C', 'B', 'B', 'physiology'),   # Muscle lactate false - OLMo wrong, Qwen correct
    125: (False, True, 'C', 'B', 'B', 'biochemistry'), # ATP resynthesize - OLMo wrong, Qwen correct
    126: (False, False, 'C', 'C', 'A', 'biochemistry'),# Electron transport - BOTH WRONG, SAME ANSWER
    127: (False, True, 'B', 'C', 'C', 'biochemistry'), # Oxygen molecules - OLMo wrong, Qwen correct
    128: (False, True, 'D', 'C', 'C', 'psychology'),   # Substance abuse - OLMo wrong, Qwen correct
    129: (False, False, 'C', 'C', 'B', 'biochemistry'),# Glycogen breakdown - BOTH WRONG, SAME ANSWER
    130: (True, False, 'D', 'C', 'D', 'psychology'),   # Wrestler weight - OLMo correct, Qwen wrong
    131: (False, False, 'C', 'A', 'B', 'biochemistry'),# SDS-PAGE - BOTH WRONG, different answers
    132: (False, False, 'C', 'C', 'D', 'physiology'),  # ATP stores - BOTH WRONG, SAME ANSWER
    133: (False, False, 'B', 'B', 'D', 'physiology'),  # Sport factors - BOTH WRONG, SAME ANSWER
    134: (False, False, 'C', 'D', 'A', 'embryology'),  # Germ layer - BOTH WRONG, different answers
    135: (False, False, 'B', 'A', 'D', 'biochemistry'),# Transmembrane - BOTH WRONG, different answers
    136: (False, True, 'C', 'A', 'A', 'genetics'),     # RNA bases - OLMo wrong, Qwen correct
    137: (True, False, 'C', 'A', 'C', 'physiology'),   # Fast-twitch fibres - OLMo correct, Qwen wrong
    138: (False, False, 'C', 'C', 'B', 'biochemistry'),# Glucose to pyruvate - BOTH WRONG, SAME ANSWER
    139: (False, False, 'C', 'C', 'B', 'hematology'),  # Acute Myeloid - BOTH WRONG, SAME ANSWER
    140: (False, False, 'B', 'A', 'C', 'physiology'),  # Lactate transport - BOTH WRONG, different answers
    141: (True, False, 'A', 'C', 'A', 'cell_biology'), # Mitosis - OLMo correct, Qwen wrong
    142: (True, True, 'C', 'C', 'C', 'sociology'),     # Social capital - BOTH CORRECT
    143: (False, False, 'B', 'B', 'A', 'immunology'),  # Passive immunity - BOTH WRONG, SAME ANSWER
    144: (True, False, 'C', 'B', 'C', 'sociology'),    # World Systems - OLMo correct, Qwen wrong
    145: (False, True, 'C', 'B', 'B', 'biochemistry'), # Kinase reactions - OLMo wrong, Qwen correct
    146: (False, True, 'C', 'D', 'D', 'physiology'),   # Lactate fate - OLMo wrong, Qwen correct
    147: (False, False, 'C', 'A', 'B', 'physiology'),  # ATP store duration - BOTH WRONG, different answers
    148: (True, False, 'D', 'A', 'D', 'physiology'),   # Glycogen activation - OLMo correct, Qwen wrong
    149: (False, False, 'C', 'B', 'A', 'psychology'),  # Attachment theory - BOTH WRONG, different answers
    150: (False, True, 'A', 'B', 'B', 'biochemistry'), # Creatine synthesis - OLMo wrong, Qwen correct
    151: (True, True, 'C', 'C', 'C', 'endocrinology'), # Hypothalamus cortisol - BOTH CORRECT
    152: (False, False, 'C', 'C', 'D', 'psychology'),  # Prejudice task force - BOTH WRONG, SAME ANSWER
    153: (False, True, 'D', 'A', 'A', 'biochemistry'), # Muscle contraction ATP - OLMo wrong, Qwen correct
    154: (True, True, 'C', 'C', 'C', 'genetics'),      # Alternative splicing - BOTH CORRECT
    155: (False, False, 'C', 'A', 'D', 'biochemistry'),# Phosphocreatine location - BOTH WRONG, different answers
    156: (False, True, 'C', 'B', 'B', 'endocrinology'),# Steroid hormone - OLMo wrong, Qwen correct
    157: (False, True, 'C', 'B', 'B', 'genetics'),     # Exons - OLMo wrong, Qwen correct
    158: (False, False, 'C', 'C', 'A', 'sociology'),   # Symbolic culture - BOTH WRONG, SAME ANSWER
    159: (False, False, 'B', 'B', 'C', 'genetics'),    # Gene expression - BOTH WRONG, SAME ANSWER
    160: (True, False, 'C', 'D', 'C', 'biochemistry'), # Anaerobic metabolism - OLMo correct, Qwen wrong
    161: (False, True, 'C', 'D', 'D', 'genetics'),     # DNA molecules - OLMo wrong, Qwen correct
    162: (True, False, 'C', 'D', 'C', 'chemistry'),    # Gold electrons - OLMo correct, Qwen wrong
    163: (False, True, 'C', 'D', 'D', 'physiology'),   # Sauna benefits - OLMo wrong, Qwen correct
    164: (False, True, 'B', 'D', 'D', 'physiology'),   # Sprints - OLMo wrong, Qwen correct
    165: (False, False, 'C', 'C', 'D', 'physics'),     # Fire hose - BOTH WRONG, SAME ANSWER
    166: (False, False, 'C', 'C', 'B', 'biochemistry'),# FADH2 NADH - BOTH WRONG, SAME ANSWER
    167: (False, True, 'C', 'B', 'B', 'physiology'),   # ATP phosphocreatine - OLMo wrong, Qwen correct
    168: (False, False, 'C', 'B', 'D', 'genetics'),    # Sex of child - BOTH WRONG, different answers
    169: (False, True, 'C', 'A', 'A', 'physiology'),   # Blood glucose exercise - OLMo wrong, Qwen correct
    170: (False, False, 'C', 'C', 'A', 'genetics'),    # DNA replication - BOTH WRONG, SAME ANSWER
    171: (False, False, 'C', 'C', 'D', 'biochemistry'),# TCA cycle false - BOTH WRONG, SAME ANSWER
    172: (True, False, 'C', 'D', 'C', 'sociology'),    # McDonaldization - OLMo correct, Qwen wrong
    173: (True, False, 'B', 'A', 'B', 'physiology'),   # Loop of Henle - OLMo correct, Qwen wrong
}

# Analysis
print("="*80)
print("MISTAKE PATTERN ANALYSIS: OLMo-2-0425-1B vs Qwen2.5-0.5B")
print("Subject: college_medicine (Questions 110-173)")
print("="*80)

# Count categories
both_correct = 0
both_wrong_same_answer = 0
both_wrong_diff_answer = 0
olmo_only_correct = 0
qwen_only_correct = 0

both_wrong_questions = []
olmo_only_wrong = []
qwen_only_wrong = []
both_correct_questions = []

for q, (olmo_c, qwen_c, olmo_a, qwen_a, correct, topic) in results.items():
    if olmo_c and qwen_c:
        both_correct += 1
        both_correct_questions.append((q, topic))
    elif not olmo_c and not qwen_c:
        if olmo_a == qwen_a:
            both_wrong_same_answer += 1
            both_wrong_questions.append((q, topic, olmo_a, correct, "same"))
        else:
            both_wrong_diff_answer += 1
            both_wrong_questions.append((q, topic, f"OLMo:{olmo_a}/Qwen:{qwen_a}", correct, "diff"))
    elif olmo_c and not qwen_c:
        olmo_only_correct += 1
        qwen_only_wrong.append((q, topic, qwen_a, correct))
    else:  # qwen correct, olmo wrong
        qwen_only_correct += 1
        olmo_only_wrong.append((q, topic, olmo_a, correct))

total = len(results)

print(f"\n📊 AGREEMENT ANALYSIS (n={total} questions)")
print("-"*60)
print(f"Both models CORRECT:           {both_correct:3d} ({both_correct/total*100:5.1f}%)")
print(f"Both models WRONG, SAME answer:{both_wrong_same_answer:3d} ({both_wrong_same_answer/total*100:5.1f}%)")
print(f"Both models WRONG, DIFF answer:{both_wrong_diff_answer:3d} ({both_wrong_diff_answer/total*100:5.1f}%)")
print(f"Only OLMo correct (Qwen wrong):{olmo_only_correct:3d} ({olmo_only_correct/total*100:5.1f}%)")
print(f"Only Qwen correct (OLMo wrong):{qwen_only_correct:3d} ({qwen_only_correct/total*100:5.1f}%)")

print(f"\n{'─'*60}")
print("KEY FINDING: Models agree on {:.1f}% of questions".format(
    (both_correct + both_wrong_same_answer) / total * 100))
print("{'─'*60}")

# Topic analysis
print("\n\n📚 TOPIC ANALYSIS: Where do both models struggle together?")
print("-"*60)

from collections import Counter
topic_counts = Counter()
topic_both_wrong = Counter()
topic_olmo_only_wrong = Counter()
topic_qwen_only_wrong = Counter()

for q, (olmo_c, qwen_c, olmo_a, qwen_a, correct, topic) in results.items():
    topic_counts[topic] += 1
    if not olmo_c and not qwen_c:
        topic_both_wrong[topic] += 1
    elif not olmo_c:
        topic_olmo_only_wrong[topic] += 1
    elif not qwen_c:
        topic_qwen_only_wrong[topic] += 1

print(f"\n{'Topic':<20} {'Total':>6} {'Both Wrong':>11} {'OLMo Only':>10} {'Qwen Only':>10}")
print("-"*60)
for topic in sorted(topic_counts.keys()):
    total_t = topic_counts[topic]
    both = topic_both_wrong[topic]
    olmo = topic_olmo_only_wrong[topic]
    qwen = topic_qwen_only_wrong[topic]
    print(f"{topic:<20} {total_t:>6} {both:>11} {olmo:>10} {qwen:>10}")

# Questions both got wrong with same answer (systematic failure)
print("\n\n🔴 SYSTEMATIC FAILURES: Both wrong with SAME answer")
print("-"*60)
print("These questions reveal shared model weaknesses:")
for q, topic, answer, correct, _ in both_wrong_questions:
    if _ == "same":
        print(f"  Q{q}: {topic:<15} - Both answered '{answer}', correct was '{correct}'")

# Pattern in wrong answers
print("\n\n🎯 WRONG ANSWER PATTERN ANALYSIS")
print("-"*60)

olmo_wrong_choices = Counter()
qwen_wrong_choices = Counter()
correct_when_wrong = Counter()

for q, (olmo_c, qwen_c, olmo_a, qwen_a, correct, topic) in results.items():
    if not olmo_c:
        olmo_wrong_choices[olmo_a] += 1
    if not qwen_c:
        qwen_wrong_choices[qwen_a] += 1
    if not olmo_c or not qwen_c:
        correct_when_wrong[correct] += 1

print("\nOLMo wrong answer distribution:")
for choice, count in sorted(olmo_wrong_choices.items()):
    print(f"  {choice}: {count} times ({count/sum(olmo_wrong_choices.values())*100:.1f}%)")

print("\nQwen wrong answer distribution:")
for choice, count in sorted(qwen_wrong_choices.items()):
    print(f"  {choice}: {count} times ({count/sum(qwen_wrong_choices.values())*100:.1f}%)")

print("\nWhen models are wrong, the correct answer was:")
for choice, count in sorted(correct_when_wrong.items()):
    print(f"  {choice}: {count} times")

# Summary
print("\n\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
1. SHARED SYSTEMATIC FAILURES:
   - {:.1f}% of questions both models got wrong with the SAME answer
   - This suggests these questions have "attractive distractors" that
     fool both models similarly, NOT random guessing

2. COMPLEMENTARY KNOWLEDGE:
   - Qwen outperforms OLMo on {}/{} questions where only one was right
   - This means Qwen has better knowledge in specific areas

3. TOPIC WEAKNESSES:
   - Biochemistry and physiology questions are particularly challenging
   - Both models struggle with detailed metabolic pathway questions
   - "False statement" questions (asking what is NOT true) are hard

4. ANSWER BIAS:
   - OLMo strongly prefers answer 'C' when wrong ({:.1f}%)
   - This may indicate position bias or "middle option" preference

5. NOT RANDOM:
   - The high agreement rate ({:.1f}%) on wrong answers indicates 
     systematic rather than random errors
   - Models make similar reasoning mistakes
""".format(
    both_wrong_same_answer/total*100,
    qwen_only_correct,
    qwen_only_correct + olmo_only_correct,
    olmo_wrong_choices['C']/sum(olmo_wrong_choices.values())*100 if 'C' in olmo_wrong_choices else 0,
    (both_correct + both_wrong_same_answer)/total*100
))
