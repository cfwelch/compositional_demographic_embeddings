

import operator, os, re

from tqdm import tqdm
from collections import defaultdict

usa = ['alabama', 'al', 'montgomery', 'birmingham', 'alaska', 'ak', 'juneau', 'anchorage', 'arizona', 'az', 'phoenix', 'arkansas', 'ar', 'littlerock', 'california', 'ca', 'sacramento', 'los angeles', 'colorado', 'co', 'denver', 'connecticut', 'ct', 'hartford', 'bridgeport', 'delaware', 'de', 'dover', 'wilmington', 'florida', 'fl', 'tallahassee', 'jacksonville', 'georgia', 'ga', 'atlanta', 'hawaii', 'hi', 'honolulu', 'idaho', 'id', 'boise', 'illinois', 'il', 'springfield', 'chicago', 'indiana', 'in', 'indianapolis', 'iowa', 'ia', 'des moines', 'kansas', 'ks', 'topeka', 'wichita', 'kentucky', 'ky', 'frankfort', 'louisville', 'louisiana', 'la', 'baton rouge', 'new orleans', 'maine', 'me', 'augusta', 'portland', 'maryland', 'md', 'annapolis', 'baltimore', 'massachusetts', 'ma', 'boston', 'michigan', 'mi', 'lansing', 'detroit', 'minnesota', 'mn', 'st paul', 'minneapolis', 'mississippi', 'ms', 'jackson', 'missouri', 'mo', 'jefferson city', 'kansas city', 'montana', 'mt', 'helena', 'billings', 'nebraska', 'ne', 'lincoln', 'omaha', 'nevada', 'nv', 'carson city', 'las vegas', 'new hampshire', 'nh', 'concord', 'manchester', 'new jersey', 'nj', 'trenton', 'newark', 'new mexico', 'nm', 'santa fe', 'albuquerque', 'nyc', 'ny', 'albany', 'new york', 'north carolina', 'nc', 'raleigh', 'charlotte', 'north dakota', 'nd', 'bismarck', 'fargo', 'ohio', 'oh', 'columbus', 'oklahoma', 'ok', 'oklahoma city', 'oregon', 'or', 'salem', 'portland', 'pennsylvania', 'pa', 'harrisburg', 'philadelphia', 'rhode island', 'ri', 'providence', 'south carolina', 'sc', 'columbia', 'charleston', 'south dakota', 'sd', 'pierre', 'sioux falls', 'tennessee', 'tn', 'nashville', 'texas', 'tx', 'austin', 'houston', 'utah', 'ut', 'salt lake city', 'vermont', 'vt', 'montpelier', 'burlington', 'virginia', 'va', 'richmond', 'virginia beach', 'washington', 'wa', 'olympia', 'seattle', 'west virginia', 'wv', 'charleston', 'wisconsin', 'wi', 'madison', 'milwaukee', 'wyoming', 'wy', 'cheyenne', 'usa', 'states', 'america', 'united states', 'cali', 'bay', 'sf', 'jersey', 'us', 'philly', 'kc', 'st louis', 'san antonio']
asia = ['afghanistan', 'armenia', 'azerbaijan', 'bahrain', 'bangladesh', 'bhutan', 'brunei', 'cambodia', 'china', 'cyprus', 'georgia', 'india', 'indonesia', 'iran', 'iraq', 'israel', 'japan', 'jordan', 'kazakhstan', 'kuwait', 'kyrgyzstan', 'laos', 'lebanon', 'malaysia', 'maldives', 'mongolia', 'myanmar', 'nepal', 'north korea', 'oman', 'pakistan', 'palestine', 'philippines', 'qatar', 'russia', 'saudi arabia', 'singapore', 'south korea', 'sri lanka', 'syria', 'taiwan', 'tajikistan', 'thailand', 'timor-leste', 'turkey', 'turkmenistan', 'united arab emirates', 'uzbekistan', 'vietnam', 'yemen', 'asia', 'tokyo', 'delhi', 'shanghai', 'beijing', 'mumbai', 'osaka', 'dhaka', 'karachi', 'kolkata', 'chongqing', 'guangzhou', 'manila', 'tianjin', 'shenzhen', 'bangalore', 'jakarta', 'chennai', 'seoul', 'bangkok', 'hyderabad', 'middle east']
oceania = ['australia', 'fiji', 'kiribati', 'marshall islands', 'micronesia', 'nauru', 'new zealand', 'palau', 'papua new guinea', 'samoa', 'solomon islands', 'tonga', 'tuvalu', 'vanuatu', 'nz', 'sydney', 'melbourne', 'brisbane', 'perth', 'auckland', 'adelaide', 'gold coast–tweed heads', 'newcastle–maitland', 'canberra–queanbeyan', 'wellington', 'christchurch', 'sunshine coast', 'port moresby', 'honolulu', 'wollongong', 'geelong', 'hamilton', 'hobart', 'townsville', 'noumea', 'suva', 'cairns', 'darwin', 'tauranga', 'papeete', 'toowoomba', 'napier-hastings', 'dunedin', 'ballarat', 'lae', 'bendigo', 'albury–wodonga', 'launceston', 'palmerston north', 'honiara', 'mackay']
uk = ['uk', 'united kingdom', 'england', 'scotland', 'northern ireland', 'wales', 'leeds', 'london', 'birmingham', 'sheffield', 'glasgow', 'greater manchester', 'yorkshire', 'liverpool', 'edinburgh', 'cardiff']
europe = ['albania', 'andorra', 'armenia', 'austria', 'azerbaijan', 'belarus', 'belgium', 'bosnia and herzegovina', 'bulgaria', 'croatia', 'cyprus', 'czech republic', 'denmark', 'estonia', 'finland', 'france', 'georgia', 'germany', 'greece', 'hungary', 'iceland', 'ireland', 'italy', 'kazakhstan', 'kosovo', 'latvia', 'liechtenstein', 'lithuania', 'luxembourg', 'malta', 'moldova', 'monaco', 'montenegro', 'netherlands', 'north macedonia', 'norway', 'poland', 'portugal', 'romania', 'russia', 'san marino', 'serbia', 'slovakia', 'slovenia', 'spain', 'sweden', 'switzerland', 'turkey', 'ukraine', 'vatican city', 'europe', 'moscow', 'st petersburg', 'berlin', 'madrid', 'roma', 'kiev', 'paris', 'bucharest', 'budapest', 'hamburg', 'minsk', 'warsaw', 'belgrade', 'vienna', 'kharkov', 'barcelona', 'novosibirsk', 'nizhny novgorod', 'milan', 'ekaterinoburg', 'munich', 'prague', 'samara', 'omsk', 'sofia', 'dnepropetrovsk', 'kazan', 'ufa', 'chelyabinsk', 'donetsk', 'naples', 'perm', 'rostov-na-donu', 'odessa', 'volgograd', 'cologne', 'turin', 'voronezh', 'krasnoyarsk', 'saratov', 'zagreb', 'zaporozhye', 'lodz', 'marseille', 'riga', 'lviv', 'athens', 'salonika', 'stockholm', 'krakow', 'valencia', 'amsterdam', 'tolyatti', 'kryvy rig', 'sevilla', 'palermo', 'ulyanovsk', 'kishinev', 'genova', 'izhevsk', 'frankfurt', 'krasnodar', 'breslau', 'yaroslave', 'khabarovsk', 'vladivostok', 'zaragoza', 'essen', 'rotterdam', 'irkutsk', 'dortmund', 'stuttgart', 'barnaul', 'vilnius', 'poznan', 'dusseldorf', 'novokuznetsk', 'lisbon', 'helsinki', 'malaga', 'bremen', 'sarajevo', 'penza', 'ryazan', 'orenburg', 'naberezhnye chelny', 'duisburg', 'lipetsk', 'hannover', 'mykolaiv', 'tula', 'oslo', 'tyumen', 'copenhagen', 'kemerovo', 'dublin', 'cluj-napoca']
africa = ['algeria', 'angola', 'benin', 'botswana', 'burkina faso', 'burundi', 'cabo verde', 'cameroon', 'central african republic', 'chad', 'comoros', 'congo', 'congo', 'cote d\'ivoire', 'djibouti', 'egypt', 'equatorial guinea', 'eritrea', 'eswatini', 'ethiopia', 'gabon', 'gambia', 'ghana', 'guinea', 'guinea-bissau', 'kenya', 'lesotho', 'liberia', 'libya', 'madagascar', 'malawi', 'mali', 'mauritania', 'mauritius', 'morocco', 'mozambique', 'namibia', 'niger', 'nigeria', 'rwanda', 'sao tome and principe', 'senegal', 'seychelles', 'sierra leone', 'somalia', 'south africa', 'south sudan', 'sudan', 'tanzania', 'togo', 'tunisia', 'uganda', 'zambia', 'zimbabwe', 'africa', 'kairo', 'lagos', 'kinshasa', 'luanda', 'khartum', 'alexandria', 'abidjan', 'johannesburg', 'addis abeba', 'daressalam', 'kano', 'kapstadt', 'douala', 'casablanca', 'dakar', 'ethekwini', 'yaounde', 'ibadan', 'ekurhuleni', 'nairobi', 'maputo', 'algier', 'tshwane', 'abuja', 'ouagadougou', 'antananarivo', 'kumasi', 'lusaka', 'tunis', 'rabat', 'port harcourt', 'mbuji-mayi', 'lubumbashi', 'accra', 'brazzaville', 'mogadischu', 'harare', 'huambo', 'bamako', 'benin-stadt', 'lome', 'fes', 'conakry', 'kampala', 'kananga', 'n’djamena', 'monrovia', 'onitsha', 'mombasa', 'nelson mandela bay']
mexico = ['mexico', 'mexico city', 'ecatepec', 'tijuana', 'puebla', 'guadalajara', 'leon', 'juarez', 'zapopan', 'monterrey', 'nezahualcoyotl', 'mexicali', 'culiacan', 'naucalpan', 'merida', 'toluca', 'chihuahua', 'queretaro', 'aguascalientes', 'acapulco', 'hermosillo', 'san luis potosi', 'morelia', 'saltillo', 'guadalupe', 'tlalnepantla de baz', 'benito juarez', 'tabasco', 'torreon', 'chimalhuacan', 'reynosa', 'tlaquepaque', 'durango', 'tuxtla gutierrez', 'veracruz', 'irapuato', 'tultitlan', 'apodaca', 'cuautitlan izcalli', 'atizapan de zaragoza', 'matamoros', 'tonala', 'celaya', 'ixtapaluca', 'ensenada', 'xalapa', 'san nicolas de los garza', 'mazatlan', 'tlajomulco de zuniga', 'ahome', 'cajeme']
southam = ['argentina', 'bolivia', 'brazil', 'brasil', 'chile', 'colombia', 'ecuador', 'guyana', 'paraguay', 'peru', 'suriname', 'uruguay', 'venezuela', 'south america', 'sao paulo', 'lima', 'bogota', 'rio de janeiro', 'santiago', 'caracas', 'buenos aires', 'salvador', 'brasilia', 'fortaleza', 'guayaquil', 'quito', 'belo horizonte', 'medellin', 'cali', 'manaus', 'curitiba', 'maracaibo', 'recife', 'santa cruz de la sierra', 'porto alegre', 'belem', 'goiania', 'cordoba', 'montevideo', 'guarulhos', 'barranquilla', 'campinas', 'barquisimeto', 'sao luis', 'sao goncalo', 'maceio', 'callao', 'rosario', 'cartagena', 'valencia', 'el alto', 'duque de caxias', 'ciudad guayana', 'natal', 'arequipa', 'campo grande', 'teresina', 'sao bernardo do campo', 'nova iguacu', 'trujillo', 'joao pessoa', 'la paz', 'santo andre', 'osasco', 'sao jose dos campos', 'la plata', 'jaboatao dos guararapes', 'cochabamba', 'ribeirao preto', 'uberlandia', 'contagem', 'sorocaba', 'mar del plata', 'aracaju', 'cucuta', 'feira de santana', 'soledad', 'puente alto', 'chiclayo', 'salta', 'san miguel de tucuman', 'maturin', 'cuenca', 'cuiaba', 'joinville', 'juiz de fora', 'londrina', 'asuncion', 'ibague', 'aparecida de goiania', 'bucaramanga', 'ananindeua', 'soacha', 'porto velho']
canada = ['canada', 'ontario', 'quebec', 'british columbia', 'alberta', 'manitoba', 'saskatchewan', 'nova scotia', 'new brunswick', 'newfoundland', 'prince edward island', 'northwest territories', 'nunavut', 'yukon', 'edmonton', 'toronto', 'whitehorse', 'calgary', 'victoria', 'vancouver', 'winnipeg', 'fredericton', 'saint john', 'st john', 'halifax', 'charlottetown', 'montreal', 'regina', 'saskatoon', 'iqaluit', 'yellowknife']

name_dict = {'usa': usa, 'asia': asia, 'oceania': oceania, 'uk': uk, 'europe': europe, 'africa': africa, 'mexico': mexico, 'southam': southam, 'canada': canada}
PARTIAL_MATCH = False

def main():
    user_locs = defaultdict(lambda: [])

    file_list = [i for i in os.listdir('../demographic') if i.startswith('locations_')]
    for file in file_list:
        with open('../demographic/' + file) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                user = tline[0]
                loc = tline[3]
                if user == '[deleted]':
                    continue

                user_locs[user].append(loc.lower())

    broken = []
    num_locs = defaultdict(lambda: 0)
    resolved = {}
    for user in tqdm(user_locs):
        # uloc_new = [1 if i in usa else 0 for i in user_locs[user]]
        for i in user_locs[user]:
            i_t = i.replace('.', '')
            if i_t.startswith('the '):
                i_t = i_t[4:]
            if i_t.endswith(' area'):
                i_t = i_t[:-5]
            i_t = re.sub('(northern|western|eastern|southern|downtown|suburbs)', '', i_t).strip()

            if i_t in asia:
                num_locs['asia'] += 1
                resolved[user] = 'asia'
                break
            if i_t in mexico:
                num_locs['mexico'] += 1
                resolved[user] = 'mexico'
                break
            elif i_t in oceania:
                num_locs['oceania'] += 1
                resolved[user] = 'oceania'
                break
            elif i_t in africa:
                num_locs['africa'] += 1
                resolved[user] = 'africa'
                break
            elif i_t in southam:
                num_locs['southam'] += 1
                resolved[user] = 'southam'
                break
            elif i_t in uk:
                num_locs['uk'] += 1
                resolved[user] = 'uk'
                break
            elif i_t in europe:
                num_locs['europe'] += 1
                resolved[user] = 'europe'
                break
            elif i_t in canada:
                num_locs['canada'] += 1
                resolved[user] = 'canada'
                break
            elif i_t in usa:
                num_locs['usa'] += 1
                resolved[user] = 'usa'
                break
            else:
                obreak = False
                if PARTIAL_MATCH:
                    # try inexact match
                    for k,v in name_dict.items():
                        for z in v:
                            if len(re.findall('(^|$| )' + z + '(^|$| )', i)):
                                num_locs[k] += 1
                                resolved[user] = k
                                obreak = True
                                break
                        if obreak:
                            break
                    if obreak:
                        break
                    else:
                        broken.append(i)
                else:
                    broken.append(i)

        # uloc_new = [i + ' --> USA' if i in usa else i for i in user_locs[user]]
        # print(user + ': ' + str(uloc_new))
        # input()

    print(broken)

    all_ppl = len(user_locs)
    covered = 0
    print('Number of People: ' + '{:,d}'.format(all_ppl) + '\n')

    for k,v in sorted(num_locs.items(), key=operator.itemgetter(1), reverse=True):
        covered += v
        print('Number of People from ' + k.upper() + ': ' + '{:,d}'.format(v))

    print('\nNumber of people not covered: ' + '{:,d}'.format(all_ppl - covered))

    with open('../demographic/resolved_locations', 'w') as handle:
        for k,v in resolved.items():
            handle.write(k + '\t' + v + '\n')

if __name__ == '__main__':
    main()
