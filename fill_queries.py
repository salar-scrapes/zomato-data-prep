#!/usr/bin/env python3
"""Fill `query` and `query_type` in eval_sample_draft.csv for the first 50 reviews.

Usage (from the project root):
    python fill_queries.py                       # data/processed/eval_sample_draft.csv
    python fill_queries.py path/to/draft.csv     # or pass a path

Rows are matched by review_id, so row order doesn't matter and re-running is safe.
The original CSV is saved once next to itself as <name>.csv.bak.

If evaluate.py expects different query_type labels, edit TYPE_RENAMES below
instead of touching the 50 entries.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "data/processed/eval_sample_draft.csv"

# Map my labels -> the labels evaluate.py accepts (leave empty if mine are fine).
# Example: {"dish_quality": "dish", "complaint_issue": "complaint"}
TYPE_RENAMES = {
    "dish_quality": "review_evidence",
    "price_value": "review_evidence",
    "occasion_group": "mixed",
    "dietary_cuisine": "profile_fact",
    "practical_info": "profile_fact",
    "complaint_issue": "review_evidence",
    "ambience_vibe": "review_evidence",
    "service_staff": "review_evidence"
}

# review_id -> (query, query_type)
ANNOTATIONS = {
    # 360 atoms restaurant and cafe (Banashankari)
    "3b1a815420b8c70f": ("Cafe in Banashankari where I can get creamy white sauce pasta, garlic bread and momos?", "dish_quality"),
    "6110a37e1006089f": ("Is there a quirky cafe with skeleton decor, colourful lighting and good music that's still affordable for a group of friends?", "ambience_vibe"),
    "9caae7d118b3efcb": ("Casual hangout in Banashankari 2nd stage where vegetarians and non-vegetarians in the same group will both find something to eat?", "dietary_cuisine"),
    "f10df9654e0db43c": ("Restaurant in Banashankari where the staff make you feel properly welcome and the dishes don't cost much?", "service_staff"),

    # Adupadi (Prestige Shantiniketan food court, Whitefield)
    "00a1e2177b88648c": ("Biryani combo with raita and salad in a Whitefield food court - are the portions big, and is the fried chicken too oily?", "dish_quality"),
    "254349268baadc24": ("Lunch choices inside the Prestige Shantiniketan food court for someone who prefers north Indian taste?", "dietary_cuisine"),
    "d5fdadf2635808ea": ("Does the food court counter on ITPL Main Road really serve alcohol, or is the listing misleading?", "practical_info"),
    "f863cb7bf80192bd": ("How do the staff at the Whitefield campus food court restaurant react when you complain that a dish is missing its main ingredient?", "complaint_issue"),

    # Amma's Pastries (HSR sector 6)
    "038678c68d801a77": ("Which bakery in HSR sector 6 is often called one of the best for cakes without being expensive?", "price_value"),
    "088a9a62da92c0a9": ("Are the white forest and chocolate cakes from this HSR bakery fresh enough to order for an office celebration?", "occasion_group"),
    "189c6ef9a010e0c7": ("Bakery near HSR sector 6 with a big range of cakes that are cheaper than other vendors and ready if you need one in a hurry?", "practical_info"),
    "afe4ed6d8ed04714": ("Which bakery in HSR fixes order mistakes quickly, like a manager bringing the right cake to your home after a mix-up?", "service_staff"),

    # Andhra Spices (Brookefield)
    "8422ea56494c14b5": ("Small Andhra meals place near Apollo Cradle on Brookefield main road where the limited menu keeps the food fresh and hot?", "dietary_cuisine"),
    "8d1b0164fbf35789": ("Where on ITPL Main Road can I get spicy Hyderabadi dum biryani at a reasonable price?", "dish_quality"),
    "dcc6abf7be179949": ("Any reports of undercooked chicken, small portions or poor cleanliness at the Andhra restaurant in Brookefield?", "complaint_issue"),

    # Barbeque Nation (Electronic City)
    "17fab8aafab65ea1": ("Unlimited grill buffet in Electronic City with prawn and mutton starters and pan kulfi for dessert?", "dish_quality"),
    "8706efcbde2cc9cd": ("Has the food quality at the Electronic City barbecue buffet dropped over the past year even though the service is still reliable?", "dish_quality"),
    "986ea50ea45f8499": ("I'm vegetarian: will a barbecue buffet in Doddathogur, Electronic City have enough to eat, or is it mostly meat?", "dietary_cuisine"),
    "ba54ed21baeb531d": ("Is the buffet in Electronic City still worth going back to if the main dishes start tasting alike but the live counters and desserts are good?", "dish_quality"),

    # Bharatkhana (Kundanhalli, Marathahalli)
    "3e1477d16903475a": ("Marathahalli restaurant around Kundanhalli with comfortable seating where you get good-quality food at an average price?", "practical_info"),
    "a2e607eea727eadd": ("Where near Marathahalli can I try Bengali food on a tight budget? Around Rs 400 covered loads of items, and there's balcony seating.", "dietary_cuisine"),
    "e42ea9ca7c35594b": ("Bengali restaurant in Bangalore that beats some of the famous ones and still charges reasonable rates?", "price_value"),
    "eebfb63a966162f5": ("Which Bengali restaurant in Marathahalli is known for its mutton, with a nice atmosphere and helpful staff?", "dish_quality"),

    # Blend - Royal Orchid Suites (Whitefield)
    "7f66cd00de108456": ("Hotel restaurant in Whitefield where the staff look after you well through a multi-day business stay, with home-style meals and a good breakfast?", "service_staff"),
    "90c753c69166aef3": ("Whitefield restaurant offering corporate lunch discounts with a spread for both veg and non-veg eaters?", "price_value"),
    "970524636b3c3f03": ("Hotel buffet in Whitefield for a small birthday dinner - which dishes are worth it and which are weak?", "occasion_group"),
    "b149af0c362b5cfa": ("Whitefield buffet that impressed a team dinner with senior management - good starters, biryani and desserts?", "occasion_group"),

    # Boho (Koramangala 5th block)
    "597ab43787ea5ec2": ("Energetic spot in Koramangala 5th block for a casual evening out with friends and okay food, but maybe not somewhere to revisit often?", "ambience_vibe"),
    "7e28593d1c4ee6b1": ("Koramangala spot with great music and polite staff - how expensive is it and how's the hygiene?", "price_value"),
    "bb3fd899d3e11990": ("Where in Koramangala can I party on a weekday, with Bollywood music, a dance floor and a ladies' offer on drinks?", "occasion_group"),
    "f21e3017d81dfc62": ("Is there a Koramangala venue with a live band and a DJ that also shows football matches on a projector?", "ambience_vibe"),

    # Cake Cafe (HSR sector 2)
    "610b413deae7cbbd": ("Freshness complaints about lychee cake orders from the cake shop in HSR sector 2?", "complaint_issue"),
    "cc5c778bc53d6abe": ("Custom-designed birthday cake in HSR Layout that matched the requested design, tastes fresh and costs less than other places?", "occasion_group"),
    "cdff1f7bab973df8": ("Choco chips cake for my son's birthday from an HSR Layout cake shop - is it tasty?", "dish_quality"),

    # Cakezone (Indiranagar)
    "298b6d6d21cb5aef": ("Pastries or cupcakes - which should I order from the Indiranagar bakery, and which cupcake flavours are best?", "dish_quality"),
    "9069843315ab0eef": ("Where in Indiranagar can I get a creamy molten choco lava cake, and is their red velvet worth ordering?", "dish_quality"),
    "cbb8fc4585a2d2de": ("Dessert jars from an Indiranagar bakery that are clearly better than the big cloud-kitchen brands?", "dish_quality"),
    "cf641128719a92ed": ("Indiranagar cake place with a wide variety for delivery, red velvet arriving fresh, and a handful of branches around the city?", "practical_info"),

    # Chai Days (100 Feet Road, Indiranagar)
    "4d48806a0dfc872e": ("Cozy tea-and-cake spot along 100 Feet Road with quick, efficient service for a chilly evening?", "ambience_vibe"),
    "5de6190d3af39b8d": ("Where in Indiranagar can I try saffron chai alongside snacks like a Mexican sausage roll or egg puff?", "dish_quality"),
    "ac216ffc091b6f8b": ("Tea cafe in Indiranagar worth returning to repeatedly for ginger masala chai, a wide snack menu and good value?", "price_value"),
    "b656f9191d41ec6e": ("Can I get my chai made without sugar at a tea place on 100 Feet Road in Indiranagar?", "dietary_cuisine"),

    # Chaitra Restaurant (Kalyan Nagar)
    "0ca9fd4e2224c6a7": ("Hygiene problems at a Kalyan Nagar restaurant where an insect turned up in a biryani delivery?", "complaint_issue"),
    "65c5fb393beb1b83": ("Budget restaurant in Kalyan Nagar with decent food - how's the packing for curries and other liquid items?", "practical_info"),
    "6bf8c9bf5ee423be": ("Do portions shrink on discounted Zomato orders at the Kalyan Nagar place? A Rs 130 paneer chilli came as a quarter plate.", "price_value"),
    "b3f525ba246bbbfe": ("Which Kalyan Nagar restaurant takes payment for delivery orders but doesn't deliver and leaves you waiting for days?", "complaint_issue"),

    # Chandus Hotel (Basaveshwara Nagar)
    "04665678e3ab780f": ("Is the eatery in Basaveshwara Nagar with dim lighting and drinkers around a suitable place to dine with family?", "occasion_group"),
    "0f295ce1958df6e4": ("Basic local non-veg joint in Basaveshwara Nagar serving tasty food at very low prices - if you can overlook the cleanliness?", "price_value"),
    "342be75aa480a868": ("Which Basaveshwara Nagar restaurant is known for terrible cleanliness and poor taste?", "complaint_issue"),
    "6825cd476a60b50b": ("Non-veg restaurant on the Basaveshwara Nagar main road with parking for bikes and cars, known for biriyani, kabab and fish fry?", "practical_info"),
}


def main():
    assert len(ANNOTATIONS) == 50, f"expected 50 annotations, found {len(ANNOTATIONS)} (duplicate review_id?)"
    assert len({q.lower() for q, _ in ANNOTATIONS.values()}) == 50, "duplicate query text"

    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path} (run from the project root, or pass the path)")

    # Read everything as text so ratings/IDs/review text round-trip unchanged.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "review_id" not in df.columns:
        sys.exit(f"No review_id column in {csv_path}; columns are {list(df.columns)}")
    if df["review_id"].duplicated().any():
        sys.exit("review_id is not unique in the CSV, so rows can't be matched safely.")

    missing = [rid for rid in ANNOTATIONS if rid not in set(df["review_id"])]
    if missing:
        sys.exit(f"{len(missing)} review_ids not found in the CSV, e.g. {missing[:3]}. Nothing written.")

    backup = csv_path.with_name(csv_path.name + ".bak")
    if not backup.exists():
        shutil.copy2(csv_path, backup)

    for col in ("query", "query_type"):
        if col not in df.columns:
            df[col] = ""

    for rid, (query, qtype) in ANNOTATIONS.items():
        mask = df["review_id"] == rid
        df.loc[mask, "query"] = query
        df.loc[mask, "query_type"] = TYPE_RENAMES.get(qtype, qtype)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    filled = df[df["query"].str.strip() != ""]
    print(f"Filled {len(ANNOTATIONS)} rows in {csv_path} "
          f"({len(filled)} of {len(df)} rows now have a query).")
    print(filled["query_type"].value_counts().to_string())
    print(f"Original saved as {backup}")


if __name__ == "__main__":
    main()