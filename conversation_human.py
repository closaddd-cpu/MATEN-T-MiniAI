# BY MATEN/T | M/ License v1.0
"""Original, introductory conversational starter templates; not harvested dialogue."""
SCENES = {
 'ckb': [
  ('سڵاو! من {m} خولەکم هەیە بۆ فێربوونی پایتۆن. چۆن دەست پێ بکەم؟', 'سڵاو! 😊 لە variable و print دەست پێبکە، پاشان هەر ڕۆژ نموونەیەکی بچووک تاقی بکەوە. {m} خولەک بۆ ئەمڕۆ دابەش بکە بۆ خوێندنەوە و مەشق.'),
  ('من دەمەوێت لە کیمیا تێبگەم. {m} خولەکم هەیە. یەکەم هەنگاو چییە؟', 'بەخێربێیت! 🧪 سەرەتا جیاوازی نێوان ئەتۆم، توخم و مولیکول فێربە. دواتر بە نموونە ڕاهێنان بکە. ئەگەر بابەتێک ئاڵۆز بوو، پرسیاری وردتر بکە.'),
  ('بۆ پاراستنی هەژمارەکانم پێشنیاری چییە؟ من {m} هەژمارم هەیە.', 'باشترین دەستپێک: وشەی نهێنی جیاواز، دوو-قۆناغی پشتڕاستکردنەوە، و نوێکردنەوەی بەرنامەکان. 🔐 بۆ هەر {m} هەژمارەکە تاقی بکەوە.'),
  ('سڵاو، ئەمڕۆ دەمەوێت پلانی خوێندن بۆ {m} خولەک دروست بکەم.', 'سڵاو! 🌿 {m} خولەک بە 3 بەش دابەش بکە: تێگەیشتن، ڕاهێنان، و پێداچوونەوە. لەکۆتاییدا ئەوە بنووسە کە فێرت بوو.'),
 ],
 'en': [
  ('Hi! I have {m} minutes to study programming. Where should I start?', 'Hi! 😊 Start with variables, conditions and functions. Use your {m} minutes to read one example and modify it yourself.'),
  ('Can you give me a safe cybersecurity checklist for {m} accounts?', 'Certainly! 🔐 Use unique passwords, enable MFA, check recovery options and review active sessions. Prioritize important accounts first.'),
  ('I am curious about chemistry. Can you suggest a {m}-minute study plan?', 'Sure! 🧪 Review atoms and molecules, work through one balancing exercise, then recap what each symbol means.'),
  ('Help me plan my learning time. I have {m} minutes today.', 'Let’s use a simple plan: study a concept, practice a tiny task and briefly summarize what worked. 🌱 Consistency matters more than one long session.'),
 ],
 'ar': [
  ('مرحبًا! لدي {m} دقيقة لتعلم البرمجة. من أين أبدأ؟', 'أهلًا! 😊 ابدأ بالمتغيرات والشروط والدوال. جرّب تعديل مثال صغير خلال وقتك المتاح.'),
  ('كيف أحمي حساباتي وعددها {m}؟', 'أنصح بكلمات مرور مختلفة، وتفعيل المصادقة متعددة العوامل، ومراجعة جلسات تسجيل الدخول. 🔐'),
  ('أريد دراسة الكيمياء لمدة {m} دقيقة. ما الخطة؟', 'لنبدأ بالذرات والجزيئات، ثم نحل تمرينًا بسيطًا ونراجع المفاهيم الأساسية. 🧪'),
  ('ساعدني في تنظيم {m} دقيقة للدراسة اليوم.', 'قسّم وقتك إلى فهم المفهوم، وتطبيق مثال، ومراجعة الأخطاء. 😊'),
 ]
}

def make_conversation(n,a,b):
    lang = ('ckb','en','ar')[n%3]
    items=SCENES[lang]
    q,r=items[(n//3)%len(items)]
    m=10+(a%101)
    extra = {
      'ckb': (f' هەروەها دەتوانیت {a} + {b} بۆم حیساب بکەیت؟', f' هەروەها {a} + {b} = {a+b}.'),
      'en': (f' Also, what is {a} + {b}?', f' And {a} + {b} = {a+b}.'),
      'ar': (f' وأيضًا، كم يساوي {a} + {b}؟', f' وأيضًا {a} + {b} = {a+b}.')
    }[lang]
    return {'question':q.format(m=m) + extra[0], 'answer':r.format(m=m) + extra[1], 'language':lang,'topic':'conversation/human/'+lang}
