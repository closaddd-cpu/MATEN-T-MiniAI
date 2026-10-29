# BY MATEN/T | Original, procedural educational dataset builder | M/ License v1.0
"""Generate deterministic, SYNTHETIC 100k examples per named JSONL shard.
Do not mistake the output for a professionally curated corpus or genuine human conversations.
No third-party datasets are downloaded, copied, or silently assigned a license.
"""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from facts import CARDS
from catalog import EXTRA_CODE, NATURAL, SCIENCE_CARDS
from conversation_human import make_conversation

CREATOR = 'MATEN/T'
LICENSE = 'M/ License v1.0'
ROOT = Path(__file__).resolve().parent
CODE_LANGS = ['c', 'c++', 'java', 'python', 'javascript', 'typescript', 'csharp', 'go', 'rust', 'kotlin',
              'swift', 'php', 'ruby', 'bash', 'powershell', 'sql', 'lua', 'dart', 'r', 'julia']
DOMAINS = ['conversation_human', 'conversation', 'cybersecurity', 'mathematics', 'physics', 'chemistry', 'biology', 'astronomy', 'ai_foundations'] + list(SCIENCE_CARDS)
CODE_LANGS += list(EXTRA_CODE)
NATURAL_LANGS = list(NATURAL)
FILES = CODE_LANGS + NATURAL_LANGS + DOMAINS

# Templates are deliberately simple and cross-language: arithmetic and parity.
# Each example is concrete and its expected result is calculated, not invented.
CODE = {
 'c': '#include <stdio.h>\nint main(void) { long a={a}, b={b}; printf("%ld\\n", {expr}); return 0; }',
 'c++': '#include <iostream>\nint main() { long a={a}, b={b}; std::cout << ({expr}) << "\\n"; }',
 'java': 'public class Main { public static void main(String[] args) { long a={a}, b={b}; System.out.println({expr}); } }',
 'python': 'a, b = {a}, {b}\nprint({expr})',
 'javascript': 'const a = {a}, b = {b};\nconsole.log({expr});',
 'typescript': 'const a: number = {a}, b: number = {b};\nconsole.log({expr});',
 'csharp': 'using System;\nclass Program { static void Main() { long a={a}, b={b}; Console.WriteLine({expr}); } }',
 'go': 'package main\nimport "fmt"\nfunc main() { a, b := int64({a}), int64({b}); fmt.Println({expr}) }',
 'rust': 'fn main() { let a: i64 = {a}; let b: i64 = {b}; println!("{}", {expr}); }',
 'kotlin': 'fun main() { val a = {a}L; val b = {b}L; println({expr}) }',
 'swift': 'import Foundation\nlet a = {a}, b = {b}\nprint({expr})',
 'php': '<?php\n$a = {a}; $b = {b}; echo {dollar_expr}, "\\n";',
 'ruby': 'a = {a}\nb = {b}\nputs({expr})',
 'bash': '#!/usr/bin/env bash\na={a}; b={b}\necho $(({expr}))',
 'powershell': '$a = {a}; $b = {b}; Write-Output ({dollar_expr})',
 'sql': 'SELECT {literal_expr} AS result;',
 'lua': 'local a, b = {a}, {b}\nprint({expr})',
 'dart': 'void main() { final int a = {a}, b = {b}; print({expr}); }',
 'r': 'a <- {a}; b <- {b}\ncat({expr}, "\\n")',
 'julia': 'a, b = {a}, {b}\nprintln({expr})',
}

OP = [('addition', '+', lambda a,b:a+b),
      ('subtraction', '-', lambda a,b:a-b),
      ('multiplication', '*', lambda a,b:a*b),
      ('remainder', '%', lambda a,b:a%b)]

PHYSICS = [
 ('Newton second law', lambda a,b:(f'A {a} kg body accelerates at {b} m/s². Find the net force.', f'F = ma = {a} × {b} = {a*b} N.')),
 ('kinetic energy', lambda a,b:(f'Find the kinetic energy of {a} kg moving at {b} m/s.', f'E = 1/2 mv² = 0.5 × {a} × {b}² = {a*b*b/2:g} J.')),
 ('Ohm law', lambda a,b:(f'Voltage is {a} V and resistance is {b} Ω. Calculate current.', f'I = V/R = {a}/{b} = {a/b:.5f} A (rounded).')),
 ('density', lambda a,b:(f'Find density for {a} grams occupying {b} cm³.', f'ρ = m/V = {a}/{b} = {a/b:.5f} g/cm³ (rounded).')),
 ('mechanical work',lambda a,b:(f'A constant force of {a} N moves an object {b} m in its direction. Find work.',f'W = Fd = {a} × {b} = {a*b} J.')),
]
CHEMISTRY = [
 ('molarity', lambda a,b:(f'{a} millimoles of solute are dissolved in {b} mL of solution. Find molarity.', f'{a} mmol/{b} mL = {a/b:.5f} mol/L (rounded).')),
 ('mass from moles',lambda a,b:(f'A substance has molar mass {b} g/mol; find the mass for {a} mol.', f'm = nM = {a} × {b} = {a*b} g.')),
 ('moles from mass',lambda a,b:(f'Substance molar mass is {b} g/mol. How many moles in {a} g?', f'n = m/M = {a}/{b} = {a/b:.5f} mol (rounded).')),
]
BIOLOGY = [
 ('population growth',lambda a,b:(f'A bacterial culture starts with {a} cells and doubles {b%8+1} times. How many cells result?', f'N = {a} × 2^{b%8+1} = {a*2**(b%8+1)} cells; assumes ideal doubling and no cell deaths.')),
 ('DNA base pairing',lambda a,b:(f'A double-stranded DNA sample has {a} A–T base pairs and {b} G–C base pairs. What is the total base-pair count?',f'{a} + {b} = {a+b} base pairs. Adenine pairs with thymine; guanine pairs with cytosine.')),
 ('microscope magnification',lambda a,b:(f'If an objective magnifies {a}× and the eyepiece {b}×, calculate total magnification.',f'{a} × {b} = {a*b}× total magnification.')),
]
ASTRONOMY = [
 ('light travel time',lambda a,b:(f'Light travels approximately 300,000 km/s. How far in {a} seconds?',f'd = vt = 300,000 × {a} = {300000*a:,} km (approximate).')),
 ('orbital-period arithmetic',lambda a,b:(f'A theoretical body completes one orbit every {b} days. How many complete orbits in {a} days?',f'Whole orbits = {a} // {b} = {a//b}. This is a simplified fixed-period exercise.')),
 ('planetary distance',lambda a,b:(f'A probe travels {a} km each day for {b} days at constant daily distance. Total distance?',f'{a} × {b} = {a*b} km; idealized constant rate.')),
]

CYBER_TASKS = [
 'rate-limit','failed-logins','patch-priority','token-expiry','least-privilege'
]

CARD_BY = {}
for card in CARDS:
    CARD_BY.setdefault(card[0].lower(),[]).append(card)

def make_record(name: str, n: int, seed: int=20261029) -> dict:
    rng = random.Random((n+1)*7919 + seed*104729 + FILES.index(name)*101)
    # a is injective for n in [0,100000) for the default count, so pairs remain distinct.
    a = 50 + ((n*7919 + seed*13) % 900_000)
    b = 2 + ((n*1543 + seed*17 + FILES.index(name)*31) % 997)
    if name in CODE and name in CODE_LANGS:
        op_name,sign,fn = OP[n%len(OP)]
        expected = fn(a,b)
        expr = f'a {sign} b'
        values = {'a':str(a), 'b':str(b), 'expr':expr if name!='r' else expr.replace('%','%%'),
                  'dollar_expr':f'$a {"-mod" if sign=="%" and name=="powershell" else sign} $b',
                  'literal_expr':f'{a} {sign} {b}'}
        import re
        response = re.sub(r'\{(a|b|expr|dollar_expr|literal_expr)\}', lambda m: values[m.group(1)], CODE[name])
        # Python % and C-like % differ for negative operands; all operands are positive by design.
        ins = [f'Write a complete {name} example that computes {op_name} of {a} and {b} and prints the result.',
               f'In {name}, show a runnable calculation: {a} {sign} {b}. Output the numerical answer.',
               f'Provide a short {name} program for {op_name} with operands {a} and {b}.'][(n//4)%3]
        response = f'```{name}\n{response}\n```\nExpected output: {expected}'
        topic = f'code/{op_name}'
    elif name in EXTRA_CODE:
        code = EXTRA_CODE[name].replace('{a}',str(a)).replace('{b}',str(b))
        flavor = ['Show a concise {lang} example evaluating {a} + {b}.',
                  'Write a {lang} program or snippet that calculates {a} + {b}.',
                  'Demonstrate integer addition in {lang} using the values {a} and {b}.'][n%3]
        ins = flavor.format(lang=name,a=a,b=b)
        if name == 'solidity':
            ins = f'Write a Solidity pure function returning {a} + {b}.'
            response = f'```solidity\n{code}\n```\nCalling result() returns {a+b}.'
        else:
            response = f'```{name}\n{code}\n```\nExpected numerical result: {a+b}.'
        topic = 'code/integer-addition'
    elif name in NATURAL:
        language, q_template, answer_template = NATURAL[name]
        operation = n % 3
        symbol = ('+','−','×')[operation]
        result = (a+b,a-b,a*b)[operation]
        ins = q_template.format(a=a,b=b,sign=symbol)
        response = answer_template.format(result=result)
        topic = f'conversation/{name}/arithmetic'
    elif name=='conversation_human':
        v=make_conversation(n,a,b)
        ins=v['question']
        response=v['answer']
        topic=v['topic']
    elif name=='conversation':
        relevant = [r for r in CARDS if r[0] in ('AI','Programming','Math','Physics','Chemistry','Biology','Cybersecurity')]
        card1 = relevant[(n*29+seed)%len(relevant)]
        card2 = relevant[(n*43+seed+17)%len(relevant)]
        mode=n%4
        if mode==0:
            ins=f'I am studying for {a%75+5} minutes. Explain these two ideas and check my arithmetic exercise {a}+{b}:\n1) {card1[2]}\n2) {card2[2]}'
            response=f'1) {card1[3]}\n2) {card2[3]}\nArithmetic: {a}+{b}={a+b}. Review each answer in your own words.'
        elif mode==1:
            ins=f'Hello, I have {a%10+1} minutes for a quick lesson. {card1[2]} Also, what is {a}-{b}?'
            response=f'Hello! {card1[3]} Also, {a}-{b}={a-b}. You can ask for an example if you want to go deeper.'
        elif mode==2:
            ins=f'Help me prepare a beginner study note. Topic {card1[1]}. {card1[2]} I have {a%60+10} minutes. Calculate {a} divided by {b}, with remainder.'
            response=f'Here is a concise note: {card1[3]} The quotient of {a}/{b} is {a//b} with remainder {a%b}. Try giving an example yourself.'
        else:
            ins=f'I am new to {card1[0].lower()}. I have learned for {a%90+1} days. {card1[2]} I also want to calculate {a} times {b}.'
            response=f'{card1[3]} {a}×{b}={a*b}. It helps to practice step by step.'
        topic='conversation/educational-dialogue'
    elif name=='cybersecurity':
        kind=CYBER_TASKS[n%len(CYBER_TASKS)]
        if kind=='rate-limit':
            threshold=b%120+20; actual=a%400
            ins=f'A public service logged {a} total requests today. One IP sent {actual} in a minute with a {threshold}/min policy. Should that IP trigger the defensive rate-limit?'
            response=f'{"Yes" if actual>threshold else "No"}. {actual} {">" if actual>threshold else "<="} {threshold}. Enforce the configured limit where appropriate; log metadata without secrets and monitor sustained abuse.'
        elif kind=='failed-logins':
            count=a%60; trigger=b%15+5
            ins=f'A system handled {a} successful sign-ins this week. A login monitor then saw {count} failed logins, threshold {trigger}. Should it alert? Name a safe follow-up.'
            response=f'{"Alert" if count>=trigger else "No threshold alert yet"}; compare {count} against {trigger}. Consider MFA, rate limits and reviewing account activity. Never publish credentials.'
        elif kind=='patch-priority':
            days=a%180+1; externally=b%2==0
            ins=f'An application serves {a} accounts; a known security fix has been available {days} days. Internet-exposed: {externally}. Recommend patch handling.'
            response='Inventory affected versions, assess exposure and exploitability, test the vendor fix, back up critical data, deploy promptly according to risk and verify remediation. Internet exposure increases urgency.'
        elif kind=='token-expiry':
            age=a%120; policy=b%90+1
            ins=f'A service has {a} issued tokens. One access token is {age} days old; policy lifetime {policy} days. Does that token need expiry review?'
            response=f'{"Yes" if age>policy else "Not on age alone"}. Age {age} days vs policy {policy} days. Revoke unused tokens, rotate compromised secrets and apply least privilege.'
        else:
            accounts=a; admins=b%30+1
            ins=f'A team has {accounts} accounts and {admins} administrator-role assignments. Outline a safe least-privilege review.'
            response='Check each role assignment against job duties, reduce unnecessary administrative rights, require MFA, document approvals, and re-review permissions regularly. The counts alone do not prove a breach.'
        topic=f'cybersecurity/{kind}'
    elif name=='mathematics':
        kind=n%7
        if kind==0: ins=f'Calculate {a} + {b}.'; response=f'{a} + {b} = {a+b}.'
        elif kind==1: ins=f'Calculate {a} - {b}.'; response=f'{a} - {b} = {a-b}.'
        elif kind==2: ins=f'Calculate {a} × {b}.'; response=f'{a} × {b} = {a*b}.'
        elif kind==3: ins=f'Divide {a} by {b}; give quotient and remainder.'; response=f'{a} = {b} × {a//b} + {a%b}; quotient {a//b}, remainder {a%b}.'
        elif kind==4: ins=f'What is {b}% of {a}?'; response=f'{a} × {b}/100 = {a*b/100:.2f}.'
        elif kind==5: ins=f'Solve x + {b} = {a}.'; response=f'Subtract {b} from both sides: x = {a-b}.'
        else: ins=f'Calculate the average of {a} and {b}.'; response=f'({a}+{b})/2 = {(a+b)/2:g}.'
        topic=f'mathematics/exercise-{kind}'
    elif name in ('physics','chemistry','biology','astronomy'):
        options=globals()[name.upper()]
        topic,fun=options[n%len(options)]
        ins,response=fun(a,b)
    elif name in SCIENCE_CARDS:
        topic, question, answer = SCIENCE_CARDS[name][n % len(SCIENCE_CARDS[name])]
        mode = (n // len(SCIENCE_CARDS[name])) % 3
        if mode == 0:
            ins = f'I am taking {a%75+15} minutes to learn {name.replace("_"," ")}. {question} Also calculate {a}+{b}.'
            response = f'{answer}\nArithmetic check: {a} + {b} = {a+b}.'
        elif mode == 1:
            ins = f'Explain this {name.replace("_"," ")} concept to a beginner: {question} For practice compute {a}−{b}.'
            response = f'{answer}\nPractice: {a} − {b} = {a-b}.'
        else:
            ins = f'Make a brief {name.replace("_"," ")} revision card: {question} If there are {a} items and {b} groups, calculate the quotient and remainder.'
            response = f'{answer}\nDivision: {a} = {b} × {a//b} + {a%b}; quotient {a//b}, remainder {a%b}.'
        topic = f'{name}/{topic}'
    else:
        items=CARD_BY.get('ai',[])
        card=items[(n*17+seed)%len(items)]
        examples=['Give one clear beginner definition and a practical next step.', 'Explain the idea without exaggerating its reliability.',
                  'Answer directly and mention one limitation.', 'Create a short revision note.']
        ins=f'I am preparing an AI lesson using {a} training examples and {b} validation examples. {card[2]} {examples[n%4]}'
        response=card[3]+f' Your example has {a+b} total records (training plus validation), but sample count alone cannot establish quality. Check held-out results before deployment.'
        topic=f'ai_foundations/{card[1]}'
    return {'id':f'{name}-{n:06d}','domain':name,'topic':topic,'instruction':ins,'response':response,
            'created_by':CREATOR,'license':LICENSE,'provenance':'MATEN/T-original-procedural-v1',
            'review_status':'synthetic-not-individually-reviewed'}

def build(count=100000, output=ROOT/'datasets', names=None, seed=20261029, force=False):
    if not 1 <= count <= 2_000_000: raise ValueError('Count must be 1..2,000,000 per domain')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    names=FILES if names is None else names
    for name in names:
        if name not in FILES: raise ValueError(f'Unknown domain: {name}')
        dest=output/(name+'.jsonl')
        if dest.exists() and not force:
            # Never pretend that a partial shard contains the requested row count.
            with dest.open('rb') as existing:
                actual=sum(chunk.count(b'\n') for chunk in iter(lambda: existing.read(8 << 20), b''))
            if actual != count:
                raise ValueError(
                    f'{dest.name} has {actual:,} rows, expected {count:,}; '
                    'use a new --out directory or --force to replace that shard.'
                )
            print(f'BY MATEN/T | reuse verified {dest.name} ({count:,} rows)')
            continue
        with dest.open('wb') as fout:
            for i in range(count):
                line=(json.dumps(make_record(name,i,seed),ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
                fout.write(line)
        print(f'BY MATEN/T | {dest.name}: {count:,} rows, {dest.stat().st_size/1048576:.1f} MiB')
    manifest={"created_by":CREATOR,"license":LICENSE,"generator":"build_datasets.py","seed":seed,
              "per_file_count":count,"total_records":count*len(names),"domains":{},
              "quality_note":"100% synthetic; many are arithmetic paraphrases, not 100k independently authored or expert-reviewed facts per file. Not evidence of multilingual proficiency.",
              "split":"train: id_index % 20 < 18; validation: == 18; test: == 19 (avoid evaluating within-parameter template variants as external generalization)."}
    for name in FILES:
        path=output/(name+'.jsonl')
        if not path.exists(): continue
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        manifest['domains'][name]={"path":path.name,"rows":count,"sha256":sha,"bytes":path.stat().st_size}
    manifest['total_records']=sum(v['rows'] for v in manifest['domains'].values())
    manifest['expected_per_file']=count
    manifest['programming_languages']=len(CODE_LANGS)
    manifest['natural_languages']=len(NATURAL_LANGS)
    manifest['other_domains']=len(DOMAINS)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='BY MATEN/T - build original procedural JSONL domains')
    p.add_argument('--count',type=int,default=100_000)
    p.add_argument('--out',default=str(ROOT/'datasets'))
    p.add_argument('--domains',nargs='*',choices=FILES,default=None)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--force',action='store_true')
    a=p.parse_args(); build(a.count,a.out,a.domains,a.seed,a.force)
