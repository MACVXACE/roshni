# Roshni: the team's written pieces

<!--
HOW THIS FILE WORKS
- Write under each "## key" heading. Do not rename the headings.
- Everything inside these comment brackets is an instruction and is ignored.
- Leave a section empty if it is not ready; the write-up keeps a visible placeholder (or our default text) until it is filled.
- Send this file back to Varun. Claude merges it into README.md and VIDEO_SCRIPT.md with the run's numbers; nobody edits README.md by hand.
- Rules for every piece: plain language; no number that is not in the results or in a source; a working link for every outside fact.
-->

## problem
<!-- OWNER: Zainab. OPTIONAL: replaces our default "The problem" text. Official template: 2-3 sentences. Limit 80 words.
Use only facts from the context file Part 4 or Addendum 2, or a new source you add under "Sources" below. -->
Pakistan has the world's highest diabetes prevalence: 31.4% of adults, about 34.5 million people, and in Pakistani studies about 29% of people with diabetes had diabetic retinopathy (9% sight-threatening). It is treatable if caught early, but screening is concentrated in eye hospitals: Pakistan has about 15 ophthalmologists per million people, mostly in cities. Travel is costly and harder for women, and in one clinic study over 70% of long-standing diabetes patients had never had a retinal check.

## next_steps
<!-- OWNER: Zainab. OPTIONAL: replaces our two default next steps. One or two concrete steps. Limit 60 words. -->
1. If access is granted, score the frozen tool once on the expert-graded Pakistani dataset registered at Al-Ibrahim Eye Hospital, Karachi (NCT06287645), re-setting the cut-off only on a separate local sample.
2. Test it camera by camera with low-cost cameras used in Pakistan, then run a costed one-clinic pilot measuring retakes, referrals and attendance.

## adoption
<!-- OWNER: Zainab (task Z4). OPTIONAL, goes in the appendix under "Who would test it first (a hypothesis, not a finding)".
Limit 120 words. Name an organisation only with a link showing it runs diabetic eye screening in Pakistan.
Say what would need to be true, and how you would test it. -->
Al-Ibrahim Eye Hospital, Karachi, runs a one-window diabetes eye facility that includes fundus photography, and is registering an expert-graded Pakistani retinal dataset (NCT06287645). For a first test to be worthwhile, three things must be true: the hospital shares a de-identified graded sample under ethics approval; its cameras export ordinary JPEG photos; and we fix our targets in writing before scoring. We would freeze Roshni, re-set the cut-off on one group of patients, score the remaining patients once, and report sensitivity, specificity, retakes and false all-clears for each camera.

## where_it_fails
<!-- OWNER: Laziza (task L1). REQUIRED. Look at results/missed_cases.png and results/missed_external.png.
For two missed eyes: which photo, and the likely reason (blur, small lesions, lesion at the edge, dark photo, a grade you would dispute).
Limit 55 words. -->
Photo 4 in results/missed_external.png (expert grade 3): we think it was missed because darkness and haze hide severe lesions in a poor-quality scan. Photo 8 (expert grade 2): we think bottom glare and tiny red lesions blending into clear vessels hid the disease.

## what_we_learned
<!-- OWNER: Laziza (task L1), with the team. REQUIRED (the guide lists its absence as a common mistake).
Two or three sentences: what surprised you (external photos, poor photos, the stress test) and what you would do differently. Limit 50 words. -->
It surprised us how sharply sensitivity dropped on new hospital cameras (56.1% versus 84.2%), while our quality check proved essential on degraded scans. Next time, we would calibrate the referral threshold on local clinic photos first and train on diverse cameras.

## video_miss_reason
<!-- OWNER: Laziza. REQUIRED for the video script. The main reason for misses, in a few words, e.g. "small lesions near the edge of the photo". Limit 12 words. -->
tiny lesions hidden by shadows, glare, and camera shifts

## ai_tools
<!-- OWNER: everyone. REQUIRED (rules: cite every AI tool). List every assistant each of you used and what for,
e.g. "Claude (Anthropic): code, research checks and write-up drafting; ChatGPT (OpenAI): ..." -->
Claude (Anthropic), https://claude.ai: code, experiment design, research checks and write-up drafting; Claude Fable 5.2 (Anthropic): redesign of the demo page

## demo_link
<!-- OWNER: Varun. REQUIRED. The GitHub Pages link to the browser demo, e.g. https://USERNAME.github.io/roshni/web/ -->
https://macvxace.github.io/roshni/web/

## repo_link
<!-- OWNER: Varun. REQUIRED. The public GitHub repository link. -->
https://github.com/MACVXACE/roshni

## urdu_corrections
<!-- OWNER: Zainab (task Z2). Not merged automatically: Claude applies it to the code.
Paste the four Urdu sentences (refer, retake, rescreen, disclaimer) with your corrections, or write "approved as is". -->
Applied in drscreen.py on 15 Nov 2026: Zainab's corrected wording for all four notes (refer, retake, rescreen, disclaimer); Urdu marked checked=True.

## new_sources
<!-- Anyone: any new source you used, one per line, as "Title: link". Claude adds them to the Sources list. -->
Added directly to README.template.md Sources: NICE MIB265 (2021); Jawa et al. 2016; Ahsan et al. 2021.
