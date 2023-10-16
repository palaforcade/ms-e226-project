# About our dataset

**Home page of the dataset**: https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2016-nsduh-2016-ds0001

**Dataset documentation**: https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2016/NSDUH-2016-datasets/NSDUH-2016-DS0001/NSDUH-2016-DS0001-info/NSDUH-2016-DS0001-info-codebook.pdf

This dataset contains 56897 observations of 2668 variables.

### Data collection and quality notice

The dataset is built from a US national survey on drug use from 2016. This is likely to induce some bias in our data :

- Although this survey is ran every year, we're looking at the data from 2016 only, and work under the assumption that the predictions and inferences that we'll be able to make from this data can still be useful today. This is disputable, because with the covid epidemic and other societal changes in the american society, we've seen a global increase in the use of drugs in the country. We can still assume that the causes and correlations that we observe in 2016 would still interesting to understand today's situation. But we will need to critically assess them with this prism when making inference about today's population.

- This data was gathered from a national survey, which means that the participants are not being clinically tested for the use of drugs. Since the consumption of most of these drugs is illegal in most of the US, we can imagine that although the respondents were told that is was completely anonymous, some of them would still underestimate their drug consumption in fear of retaliation. This is an assumption we will try to assess when analysis the data.

- Since the use of drugs often touches poorer populations, whom have fewer access to the internet and digital communication channels, there is a level of bias in the sampling of patients for this survey. This sampling bias was corrected by adding a weight for every observation, designed to increase the statistical significance of underrepresented populations. While this improves the quality of the data that we have, it still has some negative effects:
    - Artificially reduces the variance for certain characteristics
    - The dataset doesn't perfectly account for discrepancies within certain underrepresented populations