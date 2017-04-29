{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bayesian Modelling in Python\n",
    "Mark Regan\n",
    "\n",
    "---\n",
    "\n",
    "### Section 0: Introduction\n",
    "Welcome to \"Bayesian Modelling in Python\" - a tutorial for those interested in learning Bayesian statistics in Python. You can find a list of all tutorial sections on the project's [homepage](https://github.com/markdregan/Hangout-with-PyMC3).\n",
    "\n",
    "Statistics is a topic that never resonated with me throughout my years in university. The frequentist techniques that we were taught (p-values, etc.) felt contrived and ultimately, I turned my back on statistics as a topic that I wasn't interested in.\n",
    "\n",
    "That was until I stumbled upon Bayesian statistics - a branch of statistics quite different from the traditional frequentist statistics that most universities teach. I was inspired by a number of different publications, blogs & videos that I would highly recommend any newbies to Bayesian stats to begin with. They include:\n",
    "- [Doing Bayesian Data Analysis](http://www.amazon.com/Doing-Bayesian-Analysis-Second-Edition/dp/0124058884/ref=dp_ob_title_bk) by John Kruschke\n",
    "- [Python port](https://github.com/aloctavodia/Doing_Bayesian_data_analysis) of John Kruschke's examples by Osvaldo Martin\n",
    "- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) provided me with a great source of inspiration to learn Bayesian stats. In recognition of this influence, I've adopted the same visual styles as BMH.\n",
    "- [While My MCMC Gently Samples](http://twiecki.github.io/) blog by Thomas Wiecki\n",
    "- [Healthy Algorithms](http://healthyalgorithms.com/tag/pymc/) blog by Abraham Flaxman\n",
    "- [Scipy Tutorial 2014](https://github.com/fonnesbeck/scipy2014_tutorial) by Chris Fonnesbeck\n",
    "\n",
    "I created this tutorial in the hope that others find it useful and it helps them learn Bayesian techniques just like the above resources helped me. I'd welcome any corrections/comments/contributions from the community.\n",
    "\n",
    "---\n",
    "\n",
    "### Loading your Google Hangout chat data\n",
    "Throughout this tutorial, we will use a dataset containing all of my Google Hangout chat messages. I've removed the messages content and anonymized my friends' names; the rest of the dataset is unaltered.\n",
    "\n",
    "If you'd like to use your Hangout chat data whilst working through this tutorial, you can download your Google Hangout data from [Google Takeout](https://www.google.com/settings/takeout/custom/chat). The Hangout data is downloadable in JSON format. After downloading, you can replace the `hangouts.json` file in the data folder.\n",
    "\n",
    "The json file is heavily nested and contains a lot of redundant information. Some of the key fields are summarized below:\n",
    "\n",
    "| Field           | Description                                                    | Example                                      |\n",
    "|-----------------|----------------------------------------------------------------|----------------------------------------------|\n",
    "| `conversation_id` | Conversation id representing the chat thread                   | Ugw5Xrm3ZO5mzAfKB7V4AaABAQ                   |\n",
    "| `participants`    | List of participants in the chat thread                        | [Mark, Peter, John]                          |\n",
    "| `event_id`        | Id representing an event such as chat message or video hangout | 7-H0Z7-FkyB7-H0au2avdw                       |\n",
    "| `timestamp`       | Timestamp                                                      | 2014-08-15 01:54:12                          |\n",
    "| `message`         | Content of the message sent                                    | Went to the local wedding photographer today |\n",
    "| `sender`          | Sender of the message                                          | Mark Regan                                   |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import json\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn.apionly as sns\n",
    "\n",
    "from datetime import datetime\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('bmh')\n",
    "colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', \n",
    "          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The below code loads the json data and parses each message into a single row in a pandas DataFrame.\n",
    "> Note: the data/ directory is missing the hangouts.json file. You must download and add your own JSON file as described above. Alternatively, you can skip to the next section where I import an anonymized dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/mregan/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:75: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)\n"
     ]
    }
   ],
   "source": [
    "# Import json data\n",
    "with open('data/Hangouts.json') as json_file:\n",
    "    json_data = json.load(json_file)\n",
    "\n",
    "# Generate map from gaia_id to real name\n",
    "def user_name_mapping(data):\n",
    "    user_map = {'gaia_id': ''}\n",
    "    for state in data['conversation_state']:\n",
    "        participants = state['conversation_state']['conversation']['participant_data']\n",
    "        for participant in participants:\n",
    "            if 'fallback_name' in participant:\n",
    "                user_map[participant['id']['gaia_id']] = participant['fallback_name']\n",
    "\n",
    "    return user_map\n",
    "\n",
    "user_dict = user_name_mapping(json_data)\n",
    "\n",
    "# Parse data into flat list\n",
    "def fetch_messages(data):\n",
    "    messages = []\n",
    "    for state in data['conversation_state']:\n",
    "        conversation_state = state['conversation_state']\n",
    "        conversation = conversation_state['conversation']\n",
    "        conversation_id = conversation_state['conversation']['id']['id']\n",
    "        participants = conversation['participant_data']\n",
    "\n",
    "        all_participants = []\n",
    "        for participant in participants:\n",
    "            if 'fallback_name' in participant:\n",
    "                user = participant['fallback_name']\n",
    "            else:\n",
    "                # Scope to call G+ API to get name\n",
    "                user = participant['id']['gaia_id']\n",
    "            all_participants.append(user)\n",
    "            num_participants = len(all_participants)\n",
    "        \n",
    "        for event in conversation_state['event']:\n",
    "            try:\n",
    "                sender = user_dict[event['sender_id']['gaia_id']]\n",
    "            except:\n",
    "                sender = event['sender_id']['gaia_id']\n",
    "            \n",
    "            timestamp = datetime.fromtimestamp(float(float(event['timestamp'])/10**6.))\n",
    "            event_id = event['event_id']\n",
    "\n",
    "            if 'chat_message' in event:\n",
    "                content = event['chat_message']['message_content']\n",
    "                if 'segment' in content:\n",
    "                    segments = content['segment']\n",
    "                    for segment in segments:\n",
    "                        if 'text' in segment:\n",
    "                            message = segment['text']\n",
    "                            message_length = len(message)\n",
    "                            message_type = segment['type']\n",
    "                            if len(message) > 0:\n",
    "                                messages.append((conversation_id,\n",
    "                                                 event_id, \n",
    "                                                 timestamp, \n",
    "                                                 sender, \n",
    "                                                 message,\n",
    "                                                 message_length,\n",
    "                                                 all_participants,\n",
    "                                                 ', '.join(all_participants),\n",
    "                                                 num_participants,\n",
    "                                                 message_type))\n",
    "\n",
    "    messages.sort(key=lambda x: x[0])\n",
    "    return messages\n",
    "\n",
    "# Parse data into data frame\n",
    "cols = ['conversation_id', 'event_id', 'timestamp', 'sender', \n",
    "        'message', 'message_length', 'participants', 'participants_str', \n",
    "        'num_participants', 'message_type']\n",
    "\n",
    "messages = pd.DataFrame(fetch_messages(json_data), columns=cols).sort(['conversation_id', 'timestamp'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>conversation_id</th>\n",
       "      <th>event_id</th>\n",
       "      <th>timestamp</th>\n",
       "      <th>sender</th>\n",
       "      <th>message</th>\n",
       "      <th>message_length</th>\n",
       "      <th>participants</th>\n",
       "      <th>participants_str</th>\n",
       "      <th>num_participants</th>\n",
       "      <th>message_type</th>\n",
       "      <th>prev_timestamp</th>\n",
       "      <th>prev_sender</th>\n",
       "      <th>time_delay_seconds</th>\n",
       "      <th>time_delay_mins</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>year_month</th>\n",
       "      <th>is_weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Ugw5Xrm3ZO5mzAfKB7V4AaABAQ</td>\n",
       "      <td>7-H0Z7-FkyB7-HDBYj4KKh</td>\n",
       "      <td>2014-08-15 03:44:12.840015</td>\n",
       "      <td>Mark Regan</td>\n",
       "      <td>Thanks guys!!!</td>\n",
       "      <td>14</td>\n",
       "      <td>[Keir Alexander, Louise Alexander Regan, Mark ...</td>\n",
       "      <td>Keir Alexander, Louise Alexander Regan, Mark R...</td>\n",
       "      <td>3</td>\n",
       "      <td>TEXT</td>\n",
       "      <td>2014-08-15 03:44:00.781653</td>\n",
       "      <td>Keir Alexander</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4</td>\n",
       "      <td>2014-08</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               conversation_id                event_id  \\\n",
       "10  Ugw5Xrm3ZO5mzAfKB7V4AaABAQ  7-H0Z7-FkyB7-HDBYj4KKh   \n",
       "\n",
       "                    timestamp      sender         message  message_length  \\\n",
       "10 2014-08-15 03:44:12.840015  Mark Regan  Thanks guys!!!              14   \n",
       "\n",
       "                                         participants  \\\n",
       "10  [Keir Alexander, Louise Alexander Regan, Mark ...   \n",
       "\n",
       "                                     participants_str  num_participants  \\\n",
       "10  Keir Alexander, Louise Alexander Regan, Mark R...                 3   \n",
       "\n",
       "   message_type             prev_timestamp     prev_sender  \\\n",
       "10         TEXT 2014-08-15 03:44:00.781653  Keir Alexander   \n",
       "\n",
       "    time_delay_seconds  time_delay_mins  day_of_week year_month  is_weekend  \n",
       "10                12.0              1.0            4    2014-08           0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Engineer features\n",
    "messages['prev_timestamp'] = messages.groupby(['conversation_id'])['timestamp'].shift(1)\n",
    "messages['prev_sender'] = messages.groupby(['conversation_id'])['sender'].shift(1)\n",
    "\n",
    "# Exclude messages are are replies to oneself (not first reply)\n",
    "messages = messages[messages['sender'] != messages['prev_sender']]\n",
    "\n",
    "# Time delay\n",
    "messages['time_delay_seconds'] = (messages['timestamp'] - messages['prev_timestamp']).astype('timedelta64[s]')\n",
    "messages = messages[messages['time_delay_seconds'].notnull()]\n",
    "messages['time_delay_mins'] = np.ceil(messages['time_delay_seconds'].astype(int)/60.0)\n",
    "\n",
    "# Time attributes\n",
    "messages['day_of_week'] = messages['timestamp'].apply(lambda x: x.dayofweek)\n",
    "messages['year_month'] = messages['timestamp'].apply(lambda x: x.strftime(\"%Y-%m\"))\n",
    "messages['is_weekend'] = messages['day_of_week'].isin([5,6]).apply(lambda x: 1 if x == True else 0)\n",
    "\n",
    "# Limit to messages sent by me and exclude all messages between me and Alison\n",
    "messages = messages[(messages['sender'] == 'Mark Regan') & (messages['participants_str'] != 'Alison Darcy, Mark Regan')]\n",
    "\n",
    "# Remove messages not responded within 60 seconds\n",
    "# This introduces an issue by right censoring the data (might return to address)\n",
    "messages = messages[messages['time_delay_seconds'] < 60]\n",
    "\n",
    "messages.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now have a data model that we can work with more easily. The above table shows a single row in the pandas DataFrame. I'm interested in how long it takes me to respond to messages. Let's create some plots that describe my typical response times."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n6+zszLodq1evzqsdZ5555qC6nXLKKYPOR1tbG9FodFDe4c5HLBZLSr9+2W/51f2/T0pbt7GbmTdf\nwZr1yTdjLFp+D9fc/auktG3bttHY2DioHe1/+wuX3vazQXXL9nwsffzBtO0o1nW1tOOhvNtx7R8W\nJtUhFosxffp0jjnmGC677DKam5sHHieddBKnnHIKzc3NA1+m5s+fzzHHHMPs2bOT8p522mlEo1Ga\nm5uZc/kcYrHYQDtaW1uZc/mcgbyNjY184QtfSHp+c3Mzp556KmvWrKGlpWUgrb9uifnmXD6Hs88+\nuyxe54V8fVRaO/qvkfbuldz50iPc0HEXZy+4mFv/fj93vvTIwOP7i3/GxXf/PClt4dNtnL3gYm7q\n/ONAWnv3Sjo7O7nsssuS9vV6Xy8zb76CjjWrktKzfX1ccss1aduR6flYsWIF119/fVJaIT8/wrqu\nzIyGhgZ22GEHGhoauO666/K+rr70pS8xc+bMpNfpN7/5Tc444wzmzZuXlH7cccfx61//Ou92pH5+\nlPvrI107GhsbB42pz6YdnZ2d/HLpoqS0rb2vZ/36ePipvw5qxymnnDIw7vaze0/is3tP4qUnnmPn\ndX0Df39270nsx56s+NNyPrb7fknprz2zjr7nYwN/T97zQGIbY5xyyilZnY8VK1YkpRfqfHR3d9PY\n2JjX96t030uyPR/33nsvbW1tObejUJ8fW7ZsobW1Na/z8dfnOjP+vvvUS6uZefMVbOx5NSl94Z/u\npKOjIynt5ZdfprW1lbXd/0yuWwjno5if56+99hqX3HJNRt93y7kdiddV//wbid8T582bR0tLC/Pm\nzeOKK67gmGOOYebMmXR1dbFo0SKi0SiHHXYYBx10ENFolGnTpg16TRSKuXtRdpQPM2sFngXmAM8B\n73f3joTt9wGPu/u56Z6/bNmyDwKP7b///uyyy8hDy9euXUtzc3PZTo7SX7/P7j1pxEkcRrLh9Ve5\n86VHcmprOR+nQh+jMMsfah+FVug29Jcf9uye6c5DGPtIV36myvm1UC7yOUarV69m3rx5TB5/IONq\nhz7PmdrY20N790rOPPNMJk6cWFav50JfS4Uuv7e3l4ULFxKNRgf1PudaXmovxkhjrPPtyaiG13M+\nbRgtn5/FOM+57qNcjlEx6FotrGpoQzrZfDYM9bmwefNmVq1aBXDIlClTVgzKEKJcbykHwMwOAN4N\n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\nAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x120c8c190>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(10,7))\n",
    "ax = fig.add_subplot(211)\n",
    "\n",
    "order = np.sort(messages['year_month'].unique())\n",
    "sns.boxplot(x=messages['year_month'], y=messages['time_delay_seconds'], order=order, orient=\"v\", color=colors[5], linewidth=1, ax=ax)\n",
    "_ = ax.set_title('Response time distribution by month')\n",
    "_ = ax.set_xlabel('Month-Year')\n",
    "_ = ax.set_ylabel('Response time')\n",
    "_ = plt.xticks(rotation=30)\n",
    "\n",
    "ax = fig.add_subplot(212)\n",
    "plt.hist(messages['time_delay_seconds'].values, range=[0, 60], bins=60, histtype='stepfilled', color=colors[0])\n",
    "_ = ax.set_title('Response time distribution')\n",
    "_ = ax.set_xlabel('Response time (seconds)')\n",
    "_ = ax.set_ylabel('Number of messages')\n",
    "\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above plots give a monthly and an overall perspective of the length of time (in seconds) that it takes me to respond to messages. At this point I have a lot of questions that I want to ask of the data. For example:\n",
    "1. Is my response time affected by who I am talking to?\n",
    "2. Are there environmental factors that affect my response time (day of week, location, etc.)?\n",
    "3. What is the best and worst day to get in touch with me?\n",
    "\n",
    "Before we try and answer some of these questions, lets take some baby steps by estimating some parameters of a model that describes the above data. That'll make it easier for us to understand the data and inquire further.\n",
    "\n",
    "In the next section, we'll estimate parameters that describe the above distribution.\n",
    "\n",
    "#### [>> Go to the Next Section](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%201.%20Estimating%20model%20parameters.ipynb)\n",
    "\n",
    "### Export data for usage throughout tutorial"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# excluded some colums from csv output\n",
    "messages.drop(['participants', 'message', 'participants_str'], axis=1, inplace=True)\n",
    "\n",
    "# Save csv to data folder\n",
    "messages.to_csv('data/hangout_chat_data.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### References\n",
    "1. [Hangout reader](https://bitbucket.org/dotcs/hangouts-log-reader/) by Fabian Mueller"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    @font-face {\n",
       "        font-family: \"Computer Modern\";\n",
       "        src: url('http://9dbb143991406a7c655e-aa5fcb0a5a4ec34cff238a2d56ca4144.r56.cf5.rackcdn.com/cmunss.otf');\n",
       "    }\n",
       "    @font-face {\n",
       "        font-family: \"Computer Modern\";\n",
       "        font-weight: bold;\n",
       "        src: url('http://9dbb143991406a7c655e-aa5fcb0a5a4ec34cff238a2d56ca4144.r56.cf5.rackcdn.com/cmunsx.otf');\n",
       "    }\n",
       "    @font-face {\n",
       "        font-family: \"Computer Modern\";\n",
       "        font-style: oblique;\n",
       "        src: url('http://9dbb143991406a7c655e-aa5fcb0a5a4ec34cff238a2d56ca4144.r56.cf5.rackcdn.com/cmunsi.otf');\n",
       "    }\n",
       "    @font-face {\n",
       "        font-family: \"Computer Modern\";\n",
       "        font-weight: bold;\n",
       "        font-style: oblique;\n",
       "        src: url('http://9dbb143991406a7c655e-aa5fcb0a5a4ec34cff238a2d56ca4144.r56.cf5.rackcdn.com/cmunso.otf');\n",
       "    }\n",
       "    div.cell{\n",
       "        width:800px;\n",
       "        margin-left:16% !important;\n",
       "        margin-right:auto;\n",
       "    }\n",
       "    h1 {\n",
       "        font-family: Helvetica, serif;\n",
       "    }\n",
       "    h4{\n",
       "        margin-top:12px;\n",
       "        margin-bottom: 3px;\n",
       "       }\n",
       "    div.text_cell_render{\n",
       "        font-family: Computer Modern, \"Helvetica Neue\", Arial, Helvetica, Geneva, sans-serif;\n",
       "        line-height: 145%;\n",
       "        font-size: 130%;\n",
       "        width:800px;\n",
       "        margin-left:auto;\n",
       "        margin-right:auto;\n",
       "    }\n",
       "    .CodeMirror{\n",
       "            font-family: \"Source Code Pro\", source-code-pro,Consolas, monospace;\n",
       "    }\n",
       "    .prompt{\n",
       "        display: None;\n",
       "    }\n",
       "    .text_cell_render h5 {\n",
       "        font-weight: 300;\n",
       "        font-size: 22pt;\n",
       "        color: #4057A1;\n",
       "        font-style: italic;\n",
       "        margin-bottom: .5em;\n",
       "        margin-top: 0.5em;\n",
       "        display: block;\n",
       "    }\n",
       "    \n",
       "    .warning{\n",
       "        color: rgb( 240, 20, 20 )\n",
       "        }  \n",
       "</style>\n",
       "<script>\n",
       "    MathJax.Hub.Config({\n",
       "                        TeX: {\n",
       "                           extensions: [\"AMSmath.js\"]\n",
       "                           },\n",
       "                tex2jax: {\n",
       "                    inlineMath: [ ['$','$'], [\"\\\\(\",\"\\\\)\"] ],\n",
       "                    displayMath: [ ['$$','$$'], [\"\\\\[\",\"\\\\]\"] ]\n",
       "                },\n",
       "                displayAlign: 'center', // Change this to 'center' to center equations.\n",
       "                \"HTML-CSS\": {\n",
       "                    styles: {'.MathJax_Display': {\"margin\": 4}}\n",
       "                }\n",
       "        });\n",
       "</script>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Apply pretty styles\n",
    "from IPython.core.display import HTML\n",
    "\n",
    "def css_styling():\n",
    "    styles = open(\"styles/custom.css\", \"r\").read()\n",
    "    return HTML(styles)\n",
    "css_styling()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
