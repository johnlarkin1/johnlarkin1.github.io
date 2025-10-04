---
title: CryptoCraze - Google Sheets Price Update
layout: post
featured-img: cryptocurrency-meta
categories: [Crypto, Dev]
---

Text messages are fun... but an excel / Google Sheets spreadsheet is also great for organization.

The roomie ([Will J][will]) and I were curious about an organized way to automatically update current crypto metrics for an excel spreadsheet. We've got a couple of buddies who are trading and Will - being the bright Harvard stud that he is - pitched this idea to me saying that it would really help some of his friends out.

So I set out to create another cronjob that would run regularly to pull markets from once again, the CoinMarketCap API and disseminate the prices as we like. If you just want the code and you want to decipher what's going on solely from that, [here it is][code] and knock yourself out.

# Main Idea

The spreadsheet was going to have one column that lists the symbols of the cryptos that my friends are actively involved in. So the program should be able to read that column and parse all of the appropriate symbols and then make the corresponding call to `coinmarketcap` to search for the information that my buddies wanted to be displayed on a close-to-real-time level. So let's break this up into parts for clearer understanding. Here's a link to the generated html page.

# Initial Setup

Now, I'm not going to go into using the `coinmarketcap` python package - firstly, because it's simple to use and secondly, because I already did in this [blog post][crypto-notifications-post]. I suggest you check out that one if you want to kind of understand the API and what's going on.

However, the new piece of the puzzle that I was expanding on was connecting to Google Sheets, which has a great API. I wrote this program in python so the following was primarily the specific tutorial that I used to get started. You can find it [here][gsheets-python].

Again, I'm not going to go into the exact details on setting up connection for the API, but I will say that I do think it's fundamental for you to be able to get the `quickstart.py` program that Google provides. Again, see step #3 [here][gsheets-python].

Once everything is good there and you're able to successfully print out the names and majors of a dummy open spreadsheet that Google provides, then the world is yours and you can really get cracking. Writing to sheets is just another method call to the service. There's a couple of things of importance that are to highlight.

However, we do need a little bit of specific information about the spreadsheet that you're trying to work with. Google in their quickstart program uses a [dummy excel spreadsheet][dummy-sheet]. You can find the actual program [here][g-quickstart-prog]. A couple things about the terminology that the API tutorial uses can be found [here][gsheets-concepts]. But I'll cover them quickly below. Let's look at the url of the dummy sheet.

```
https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=0
```

This can be broken into a couple of main parts

The `id` portion of this spreadsheet is everything between the `d/` and the `/edit` bit. That means this spreadsheet id is

```
1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms
```

The `sheetId` portion of the spreadsheet is what comes after the `edit#gid=` bit. So this has a sheetId of

```
0
```

Those are really the two important things you'll need.

So the way I arranged the program was to pull from a yaml. The yaml file looks like

```
spreadsheet_id: <insert yours here>
sheet_id: <insert yours here>
update_time_range_name: <insert yours here>
to_write_range_name: <insert yours here>
crypto_ticker_range_name: <insert yours here>
value_input_option: <either RAW or USER_ENTERED>
```

I pushed up a dummy one with the commit as well. Also note, that Google designed the program to work that for the first time you actually log into the spreadsheet it will create a `client_secret.json` file and populate it with various params needed for the connection. After you log in once, you should be good to go.

For my yaml dictionary, we already talked about the ids. The range name is specified in A1 notation. There's three ranges total. One range to specify the time you updated the spreadsheet (a helpful indication of if it was successfully done by your cronjob). One range is to pull the crypto symbols to specify what you care about. Then a final range so that you're able to write all of the information you want into that reason. Anyway, let's expand a bit further.

# Reading the Data

The actual code that reqads the data and parses the tickert is as simple as this.

```python
result = service.spreadsheets().values().get(
    spreadsheetId=spreadsheet_id, range=crypto_symbol_range_name).execute()
# Saying if values doesn't exist, return [] as default value. Values is a list of lists.
# List per every row.
read_values = result.get('values', [])
flat_list_of_symbols = [item for sublist in read_values for item in sublist]
```

Note a couple of things. We had to flatten the list of lists to make the pertinent data more easily readable. This is because you get a list per every row that is read.

Secondly, the call to read is really simple. It's really just paying close attention to the format that everything is in so that the reading and writing is done cleanly.

# Writing the Data

This was a bit more of the fun part. So from above, we have our flat_list_of_symbols. So this is going to be like:

```
Flat list of symbols: [u'BTC', u'ETH', u'MIOTA', u'XRP]
```

So that's bitcoin, ethereum, iota, and ripple for example. A random example of what one might be interested in.

We've got this list and we want to get let's say the prices, the marketcap, and percent change for the past 24 hours since the time of the query. Luckily for us, this information is all provided by the CoinMarketCap API.

We run into an issue here. Mainly, the size of the data that we care about. We don't want to truncate the amount of market data that we've actually grabbed because who knows - maybe my lunatic friends are betting big on some small crypto that just barely crept into the top 100 in terms of market cap. But at the same time, we don't want to have to iterate over this entire 100 length list of dictionaries to find the crypto information that we care about. The final thing is that order does matter here. We received the crypto symbols in an order from top to bottom in terms of the column. We want to write the correct price, market cap, and percent change that corresponds with the right crypto. So we need to ensure that we're at least iterating over the flattened list of symbols in order.

So that's fine, but then I didn't think it would be optimal to have a double for loop with the inner loop iterating once every 100 times. This means the big O of this specific step would be $$ O(100 * n) $$ where $$ n $$ is the number of cryptos that my buddies are invested in. *Technically*, that's still linear because the size of the inner loop is fixed, but we're not dealing with theory of comp here. We're dealing with actual time. So I figured it'd be a lot better to filter our list of dictionaries to only contain relevant cryptos. Again, now it's *theoretically* $$ O(n^2) $$ but *in reality\* (which is what we live in) this is going to be optimized to the first approach. One other step that we can take to optimize the runtime of the loop is to break as soon as we find the right values to append. The other thing we need to be cautious of is - ok well what's the runtime of the filter operation? I'm guessing linear, but it's also written in C or C++ backend, so I'll take that essentially anyday.

Anyway, the final result for grabbing the crypto price, market cap, and percent change for the past 24 hours is just these clean pythonic couple of lines.

```python
filt_crypto_market_info = list(filter(lambda d: d['symbol'] in flat_list_of_symbols, crypto_market_info))
for crypto_symbol in list_of_symbols:
    for crypto_info in crypto_market_info:
        if crypto_symbol == crypto_info['symbol']:
            values_to_write.append([crypto_info['price_usd'], crypto_info['market_cap_usd'], crypto_info['percent_change_24h'] + '%']) # need to add % for excel
            break
```

# Formatting the Data Correctly

Again, another slightly tricky part of this program was the fluctuation between the google sheets API call. If you're doing multiple updates per one batch call you need to format the data differently than say if you were to only update one range of cells. Otherwise, the google sheets API will complain about it. Specifically, for the way I'm doing it, we need to store the data we want to write as a value in a dictionary that we pass to the API call. The type of the data is a list of dictionaries that correspond to the values to write for each block. So it's decently nested which can make debugging slightly tricky.

The writing portion is pretty much handled my the API call which makes this really nice and clean. So I'm not going to cover anything about that.

# Emailing the Spreadsheet After Update

Also because I figured that it was too much for my friends to actually check the spreadsheet on a mobile device or like text me and ask if I've updated it yet, I set up a cronjob and also wrote a google sheets function which was a first for me.

What you can do is go to `Tools > Script editor... > Create new project` and then pretty much just go to town. You can also automate when this new project is called, so I just did twice a day. The google scripting language is very similar to javascript, which I have a very limited proficiency in but I was able to get the job done here.

Here's the code that builds the email and sends a saved pdf that has all the information of the spreadsheet. Rather than generated text, the pdf actually preserves format and highlighting (and it's already there) so I figured I'd use that.

```javascript
function createAndSendPDF() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var documentProperties = PropertiesService.getDocumentProperties();
  var users = SpreadsheetApp.getActiveSpreadsheet().getEditors();
  var emailBuilder = "";

  // Build up the emails
  for (var i = 0; i < users.length; i++) {
    var emailBuilder = emailBuilder.concat(users[i].getEmail());
    var emailBuilder = emailBuilder.concat(",");
  }

  // Trim off excess comma
  var emailBuilder = emailBuilder.substring(0, emailBuilder.length - 1);

  // This is going to be the message in the email
  var messageInEmail =
    "Please see the attached for your daily crypto report #hodl";
  var subjectOfEmail = "Daily CryptoCM Update";

  // Generate pdf
  var pdf = DriveApp.getFileById(spreadsheet.getId())
    .getAs("application/pdf")
    .getBytes();
  var attach = {
    fileName: "Daily CryptoCM Report.pdf",
    content: pdf,
    mimeType: "application/pdf",
  };

  // Send the freshly constructed email
  MailApp.sendEmail(emailBuilder, subjectOfEmail, messageInEmail, {
    attachments: [attach],
  });
}
```

# Generating Documentation

One other thing that I dipped my toes into with this project is pydoc and the ability to use pydoc style comments to generate a helpful html page just explaining the script and some of the function calls. Really, all you need to do is have:

```
pydoc -w <name_of_python_file_without_extension>
```

and the html file will get created automatically. Also, it's important to note that you need to design your comments in a certain way in order for the parser to pick them up when the html doc is being created. I can't really link it but if you pull it down from Github you can see what it would actually look like in a browser.

# Finalizing the Product

A good confirmation check is just running the script and making sure the updated time in your spreadsheet is recent and that all of the crypto metrics look correct. Another good check program side would be ensuring that the number of updated cells that is logged is what you want.

After that you're good to go with setting up that cronjob to regularly pull in these prices if your friends want to be up to date with their returns and losses and current status of their portfolio. I'm not going to go into that detail with this post, but you can check out my other post [here][crypto-notifications-post].

My typical comments at the end. Let me know if you think I can improve my code algorithmically, design-wise, better comments or whatever. I always appreciate the feedback. Thanks for reading as always. [Here's the code][code] if you didn't already peek it!

[comment]: <> (Bibliography)
[will]: https://www.linkedin.com/in/joseph-willcox-will-jaroszewicz-27313466
[crypto-notifications-post]: https://johnlarkin1.github.io/2017/12/12/crypto-notifications.html
[gsheets-python]: https://developers.google.com/sheets/api/quickstart/python
[g-quickstart-prog]: https://developers.google.com/sheets/api/quickstart/python
[dummy-sheet]: https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
[gsheets-concepts]: https://developers.google.com/sheets/api/guides/concepts
[code]: https://github.com/johnlarkin1/crypto-sheets-update
