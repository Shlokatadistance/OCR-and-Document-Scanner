
from googlesearch import search
import time
import requests
from bs4 import BeautifulSoup
import smtplib
import urllib
import certifi

a = int(input())
#the below is a test url which did help me a lot while testing this code
#URL = "https://www.amazon.in/Razer-Blade-15-RTX-Smallest/dp/B07MK77VXZ/ref=sr_1_3?keywords=Razer+blade&qid=1581608276&sr=8-3"
#browser informations
# this is system specific information, get it on google, put some effort into knowing what it is
headers = {"user-Agents": 'your-user-agent'}
#enter two priduct links, gives you the output, on the price which has dropped

def pricecomparator():
    for i in range(a):
        i = input()
        #checkprice(i)
        if checkprice(i):
            print("the email was sent")
        else:
            print("the email wasn't sent")
            alternate_searches()
def send_mail(URL):
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    #ehlo is a command sent by an email server, estabilishes a connection
    server.starttls()
    server.ehlo()
    #below is an email specific password, generated specially for this system
    server.login('source-email-id','specific password')
    subject = "Price finally went down people"
    body = URL
    msg = f"Subject:{subject}\n\n{body}"
    server.sendmail('source-email-id','destination-email-id',msg)

    print("email sent")
    return True

    server.quit()


def checkprice(URL):

    page = requests.get(URL,headers = headers)
    soup = BeautifulSoup(page.content,'html.parser')
    #this step is pulling out data from the html page, like product title will be alogn
    #along the html script with the page
    title = soup.find(id ="productTitle").get_text()
    price = soup.find(id="priceblock_ourprice").get_text()
    #print(price)
    #converting the number to a string,checks the first five(which is the price)
    print(title.strip())

    converted_price = int(price[2:3])
    #amazon of different countries can present their prices in different formats, so the conversion into the integer using concatenation
    #may not be the best option,hence i make use of the len of string method. It is not completely correct to use it here, however it does
    #present us with some intital working methods
    if (converted_price) > 1000:
        send_mail(URL)
        return True
        #help me in judging the statements in the price comaparator
    else:
        print("nothing happens")
        #alternate_searches()
        return  False
#suppose you do not get the results you want to, you can use the below functions
#to get an alternative set of links, and you can later on try these
def alternate_searches():
    x = input()
    for j in search(x,tld="com",lang = 'en', num = 5,stop=1,pause=2):
        print(j)
#runs for a specific period
while(True):

    pricecomparator()
    time.sleep(128000)


