## Level 0

**Prompt:**

Available tool definitions:
quote(material: str in {steel, wood, glass, copper}, total: int > 0, rush: bool default: False)
rent(vehicle: str in {van, sedan, truck, scooter}, total: int > 0, insurance: bool default: False)

Request: rent a sedan. It is priced at $1 for each of 2 units, plus a flat $0 base fee. Use full insurance. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

rent(vehicle=sedan, total=2, insurance=True)

**Prompt:**

Available tool definitions:
quote(material: str in {steel, wood, glass, copper}, total: int > 0, rush: bool default: False)
ship(destination: str in {London, Paris, Tokyo, Miami}, total: int > 0, express: bool default: False)

Request: ship a package to Tokyo. It is priced at $3 for each of 3 units, plus a flat $0 base fee. Use standard delivery. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

ship(destination=Tokyo, total=9, express=False)

## Level 2

**Prompt:**

Available tool definitions:
import(origin: str in {Delhi, Hanoi, Lagos, Lima}, total: int > 0, customs: bool default: False)
rent(vehicle: str in {van, sedan, truck, scooter}, total: int > 0, insurance: bool default: False)
ship(destination: str in {London, Paris, Tokyo, Miami}, total: int > 0, express: bool default: False)

Request: import goods from Hanoi. It is priced at $7 for each of 9 units, plus a flat $0 base fee. A $4 service surcharge also applies. Use expedited customs. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

import(origin=Hanoi, total=67, customs=True)

**Prompt:**

Available tool definitions:
cater(menu: str in {pizza, sushi, bbq, vegan}, total: int > 0, gratuity: bool default: False)
rent(vehicle: str in {van, sedan, truck, scooter}, total: int > 0, insurance: bool default: False)
ship(destination: str in {London, Paris, Tokyo, Miami}, total: int > 0, express: bool default: False)

Request: cater with vegan. It is priced at $2 for each of 6 units, plus a flat $9 base fee. Use gratuity excluded. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

cater(menu=vegan, total=21, gratuity=False)

## Level 5

**Prompt:**

Available tool definitions:
order(product: str in {paper, pens, notebooks, desks}, total: int > 0, gift: bool default: False)
quote(material: str in {steel, wood, glass, copper}, total: int > 0, rush: bool default: False)
rent(vehicle: str in {van, sedan, truck, scooter}, total: int > 0, insurance: bool default: False)
ship(destination: str in {London, Paris, Tokyo, Miami}, total: int > 0, express: bool default: False)

Request: produce a quote for wood. It is priced at $23 for each of 6 units, plus a flat $8 base fee. Use standard turnaround. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

quote(material=wood, total=146, rush=False)

**Prompt:**

Available tool definitions:
cater(menu: str in {pizza, sushi, bbq, vegan}, total: int > 0, gratuity: bool default: False)
order(product: str in {paper, pens, notebooks, desks}, total: int > 0, gift: bool default: False)
quote(material: str in {steel, wood, glass, copper}, total: int > 0, rush: bool default: False)
ship(destination: str in {London, Paris, Tokyo, Miami}, total: int > 0, express: bool default: False)

Request: order paper. It is priced at $9 for each of 7 units, plus a flat $3 base fee. A $15 service surcharge also applies. Use plain packaging. Determine the total.

Call the matching tool with all of its arguments. Write the answer exactly as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, numbers as integers, and text as its plain value (for example: ship(destination=Tokyo, total=132, express=True)).

**Answer:**

order(product=paper, total=81, gift=False)

