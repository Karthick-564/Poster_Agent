"""Write the evaluation briefs (eval/briefs/*.yaml): fictional events across the genres the agent must handle. No institution assets.
Run once; the files are the evaluation set (edit them by hand to extend it)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "briefs"

BRIEFS = {
    "workshop_data": dict(format="story", event_title="DATA ANALYTICS WORKSHOP", tagline="From spreadsheets to insight",
                          subtitle="Department of Information Technology", date="8 & 9 FEBRUARY 2027", time="9:30 AM TO 4:30 PM",
                          venue="COMPUTER LAB 3", extra_lines=["DAY 1: CLEANING AND VISUALISING DATA", "DAY 2: PREDICTIVE MODELS"],
                          cta="Register before 1 February", contact_lines=["Workshop desk: analytics@example.edu"],
                          style_notes="Modern and clear. A calm generated background is fine."),
    "placement_drive": dict(format="story", event_title="CAMPUS PLACEMENT DRIVE 2027", tagline="Your career starts here",
                            date="22 FEBRUARY 2027", time="9:00 AM ONWARDS", venue="ADMINISTRATIVE BLOCK",
                            extra_lines=["ELIGIBLE: FINAL YEAR B.E. AND B.TECH", "BRING YOUR RESUME AND ID CARD"],
                            cta="Register at the placement cell", contact_lines=["Placement Cell"],
                            style_notes="Confident and energetic, professional."),
    "conference_ai": dict(format="story", event_title="INTERNATIONAL CONFERENCE ON ARTIFICIAL INTELLIGENCE", tagline="ICAI 2027",
                          date="18 - 20 MARCH 2027", venue="CONVENTION CENTRE", extra_lines=["KEYNOTES", "PAPER SESSIONS", "INDUSTRY TRACK"],
                          cta="Submit papers by 15 January", contact_lines=["icai2027@example.edu"],
                          style_notes="Academic and international. An abstract generated background is welcome."),
    "talk_climate": dict(format="story", event_title="CLIMATE AND THE CAMPUS", tagline="A public lecture",
                         subtitle="Centre for Environmental Studies", date="FRIDAY, 12 MARCH 2027", time="3:00 PM", venue="MAIN AUDITORIUM",
                         extra_lines=["CHIEF GUEST: DR. A. RAMANATHAN", "PROFESSOR OF CLIMATE SCIENCE"], contact_lines=["Entry is free"],
                         style_notes="Calm and fresh. A soft natural generated background."),
    "cultural_fest": dict(format="story", event_title="UTSAV 2027", tagline="The annual cultural festival", subtitle="Cultural Committee",
                          date="3 - 5 APRIL 2027", time="FROM 5:00 PM EACH EVENING", venue="OPEN AIR THEATRE",
                          extra_lines=["DANCE", "MUSIC", "DRAMA", "FASHION SHOW"], contact_lines=["Passes at the Student Council Office"],
                          style_notes="Festive and warm but still institutional. Generated background with warm light."),
    "sports_day": dict(format="story", event_title="ANNUAL SPORTS DAY", tagline="Run. Jump. Win.", date="15 JANUARY 2027",
                       time="8:00 AM ONWARDS", venue="COLLEGE SPORTS GROUND",
                       extra_lines=["TRACK EVENTS", "FIELD EVENTS", "RELAY RACES", "TUG OF WAR"], cta="Team entries close 8 January",
                       contact_lines=["Physical Education Department"],
                       style_notes="Energetic, bold. A generated background of a running track from above is welcome."),
    "admissions_open": dict(format="story", event_title="ADMISSIONS OPEN 2027", tagline="Engineering and Management programmes",
                            extra_lines=["B.E. COMPUTER SCIENCE", "B.E. MECHANICAL", "B.E. ELECTRONICS", "MBA"],
                            cta="Apply online before 30 May", contact_lines=["admissions@example.edu"],
                            style_notes="Clean and trustworthy. Supplied photos are not available."),
    "hackathon": dict(format="fourfive", event_title="CODE FORGE 24-HOUR HACKATHON", tagline="Build something that matters",
                      date="10 & 11 APRIL 2027", venue="INNOVATION CENTRE", extra_lines=["TEAMS OF 2 TO 4", "CASH PRIZES FOR TOP 3"],
                      cta="Register your team by 5 April", contact_lines=["Coding Club"], style_notes="Techy, bold, dark."),
    "alumni_meet": dict(format="fourfive", event_title="ALUMNI HOMECOMING", tagline="Come back to where it began",
                        date="SATURDAY, 26 JUNE 2027", time="10:00 AM", venue="COLLEGE QUADRANGLE",
                        extra_lines=["BATCHES 1990 TO 2020", "DINNER AND CULTURAL EVENING"], contact_lines=["alumni@example.edu"],
                        style_notes="Warm and nostalgic, still professional."),
    "seminar_energy": dict(format="fourfive", event_title="NATIONAL SEMINAR ON RENEWABLE ENERGY", date="25 MAY 2027", time="10:00 AM",
                           venue="SEMINAR HALL 1", extra_lines=["SOLAR AND WIND", "ENERGY STORAGE", "GRID INTEGRATION"],
                           contact_lines=["Department of Electrical Engineering"], style_notes="Technical and clean."),
    "convocation": dict(format="story", event_title="CONVOCATION 2027", tagline="Graduating class of 2023 to 2027",
                        date="SATURDAY, 7 AUGUST 2027", time="10:00 AM", venue="COLLEGE AUDITORIUM", extra_lines=["CHIEF GUEST: DR. S. KRISHNAN"],
                        style_notes="Formal and dignified. Restrained."),
    "product_launch": dict(format="story", event_title="LAUNCH OF THE INNOVATION LAB", tagline="Where ideas become prototypes",
                           date="1 SEPTEMBER 2027", time="11:00 AM", venue="ENGINEERING BLOCK, GROUND FLOOR",
                           contact_lines=["Open to all students"], style_notes="Modern, optimistic. Generated abstract background welcome."),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, b in BRIEFS.items():
        b = {"brand": "kct", **b}
        (OUT / f"{name}.yaml").write_text(yaml.safe_dump(b, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{len(BRIEFS)} briefs -> {OUT}")


if __name__ == "__main__":
    main()
