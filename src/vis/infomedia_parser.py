#!/home/knielbo/virtenvs/teki/bin/python
"""

# front page only for us
$ python infomedia_parser.py --dataset ../dat/NEWS-DATA/berglinske-print --pagecontrol True --page 1 --sort True --verbose 10

# all pages for Peter
$ python infomedia_parser.py --dataset ../dat/NEWS-DATA/berglinske-print --pagecontrol False --page 1 --sort True --verbose 10
"""
import os
import argparse
import json
import re
import glob
import newlinejson


def preprocess(dobj):
     # filters
    stopwords = [r"forsidehenvisning", r" side "]#, r"side", r"SIDE"]
    pat0 = re.compile(r"<.*?>")# remove html tags
    pat1 = re.compile(r" +")# remove extra spacing to deal with p1 header


    text = dobj["BodyText"]
    heading = dobj["Heading"]
    subheading = dobj["SubHeading"]
    text = text + " " + heading + " " + subheading


    text = re.sub(pat0, " ", text)
    for word in stopwords:
        text = re.sub(word, " ", text, flags=re.IGNORECASE)
    
    text = re.sub(pat1, " ", text)    
    title = dobj["Paragraph"]
    date = dobj["PublishDate"]

    return text, title, date

flatten = lambda l: [item for sublist in l for item in sublist]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to folder with input data")
    ap.add_argument("-c", "--pagecontrol", required=False, default=False, help="if extraction should be focused on specific page")
    ap.add_argument("-p", "--page", required=False, type=int, default=1, help="which page to be focused on, default is front page")
    ap.add_argument("-s", "--sort", required=False, type=bool, default=True, help="sort data in date")
    ap.add_argument("-v", "--verbose", required=False, type=int, default=-1, help="verbose mode (number of object to print), -1 to deactivate")
    ap.add_argument('-fn', '--filename', required=False, type=bool, default=False, help='Print filenames during processing')
    args = vars(ap.parse_args())

    TEXT, TITLE, DATE = list(), list(), list()
    error = list()

    filenames = glob.glob(os.path.join(args["dataset"], "*.ndjson"))

    for i, filename in enumerate(filenames):

        if args['filename']:
            print(filename)

        with open(filename, "r") as fobj:
            lignes = fobj.readlines()
            if lignes:
                texts = list()
                titles = list()
                dates = list()
      
                for ligne in lignes:
                    dobj = json.loads(ligne)
                    # control for missing PageIds
                    if dobj["PageIds"][0]:
                        pageid = int(dobj["PageIds"][0])
                    else:
                        pageid = 'NA (PageIds blank in API)'
                    # extract date from page
                    if args["pagecontrol"]:
                        if pageid == args["page"]:
                            text, title, date = preprocess(dobj)
                            texts.append(text)
                            titles.append(title)
                            dates.append(date)
                    # get all data
                    else:
                        text, title, date = preprocess(dobj)
                        texts.append(text)
                        titles.append(title)
                        dates.append(date)
                
                # concatenate all content on page 
                if args["pagecontrol"]:
                    # control for empty pages
                    if texts and dates and titles:
                        texts = [" ".join(texts)]
                        dates = [dates[0]]
                        titles = [" ".join(titles)]
                    else:
                        texts = []
                        dates = []
                        titles = []

                TEXT.append(texts)
                DATE.append(dates)
                TITLE.append(titles)

            # record empty files
            else:
                error.append(os.path.basename(filename))

            if args["verbose"] > 0 and i > 0 and (i + 1) % args["verbose"] == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(filenames)))

    print("[INFO] {} of {} are empty in {} ...".format(len(error),len(filenames), os.path.basename(args["dataset"])))
        
    # flatten ls of ls
    TEXT = flatten(TEXT)
    TITLE = flatten(TITLE)
    DATE = flatten(DATE)
    
    # sort data on date
    if args["sort"]:
        TEXT = [text for _,text in sorted(zip(DATE, TEXT))]
        TITLE = [title for _,title in sorted(zip(DATE, TITLE))]
        DATE = sorted(DATE)

    # write to external
    lignes = list()
    for i, date in enumerate(DATE):
        d = dict()
        d["date"] = date
        d["text"] = TEXT[i]
        d["title"] = TITLE[i]
        lignes.append(d)

    # folder
    if args['pagecontrol']:
        outdir = 'FrontPage'
    else:
        outdir = 'AllPages'
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fname = os.path.join(outdir,
                         os.path.basename(
                             os.path.normpath(args["dataset"])
                         ) + ".ndjson")

    print("[INFO] writing target data to: {}".format(fname))

    with open(fname, "w") as f:
        newlinejson.dump(lignes, f, ensure_ascii=False)

if __name__=="__main__":
    main()