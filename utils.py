

import datetime, socket, base36, time, html, pytz, re, os

from termcolor import colored

dir_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(dir_path + "/logs", exist_ok=True)

# Data folder names in ./data/ assuming structure ./data/YYYY/RC_YYYY-MM
DIR_SET = ['2007', '2008', '2009']#, '2010', '2011', '2012', '2013', '2014', '2015']

# Constants for learning compositional embeddings

AGES = ['young', 'old', 'unknown']
LOCATIONS = ['usa', 'asia', 'oceania', 'uk', 'europe', 'canada', 'unknown'] # 'africa', 'mexico', 'southam'
RELIGIONS = ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'unknown']
GENDERS = ['male', 'female', 'unknown']

# Constants for personalized embeddings

liwc_keys = ['ACHIEV', 'ADJ', 'ADVERB', 'AFFECT', 'AFFILIATION', 'ANGER', 'ANX', 'ARTICLE', 'ASSENT', 'AUXVERB', 'BIO', 'BODY', 'CAUSE', 'CERTAIN', 'COGPROC', 'COMPARE', 'CONJ', 'DEATH', 'DIFFER', 'DISCREP', 'DRIVES', 'FAMILY', 'FEEL', 'FEMALE', 'FILLER', 'FOCUSFUTURE', 'FOCUSPAST', 'FOCUSPRESENT', 'FRIEND', 'FUNCTION', 'HEALTH', 'HEAR', 'HOME', 'I', 'INFORMAL', 'INGEST', 'INSIGHT', 'INTERROG', 'IPRON', 'LEISURE', 'MALE', 'MONEY', 'MOTION', 'NEGATE', 'NEGEMO', 'NETSPEAK', 'NONFLU', 'NUMBER', 'PERCEPT', 'POSEMO', 'POWER', 'PPRON', 'PREP', 'PRONOUN', 'QUANT', 'RELATIV', 'RELIG', 'REWARD', 'RISK', 'SAD', 'SEE', 'SEXUAL', 'SHEHE', 'SOCIAL', 'SPACE', 'SWEAR', 'TENTAT', 'THEY', 'TIME', 'VERB', 'WE', 'WORK', 'YOU']
roget_keys = ['RELIGIONS CULTS SECTS', 'STORE SUPPLY', 'KNOWLEDGE', 'OTHER SPORTS', 'AGE', 'UNCERTAINTY', 'BORROWING', 'LIFE', 'VISIBILITY', 'DIVORCE WIDOWHOOD', 'ENVIRONMENT', 'ACCOUNTS', 'FRAGRANCE', 'DEMAND', 'NOTCH', 'DISPROOF', 'PUBLIC SPIRIT', 'NO QUALIFICATIONS', 'PROSPERITY', 'DIRECTION', 'CONVOLUTION', 'ARTIST', 'DANCE', 'LEISURE', 'INCOMBUSTIBILITY', 'QUIESCENCE', 'ENGINEERING', 'PRINTING', 'SILENCE', 'BEHAVIOR', 'ACTION', 'COMPACT', 'PREARRANGEMENT', 'WEAVING', 'RESCUE', 'UNWILLINGNESS', 'PUBLIC SPEAKING', 'UNRELATEDNESS', 'PEACE', 'PERIOD', 'PREDICTION', 'RESIGNATION RETIREMENT', 'UNCLOTHING', 'THE PEOPLE', 'GREENNESS', 'STRIDENCY', 'COUNTERACTION', 'WIT HUMOR', 'COMPLETENESS', 'UNPLEASANTNESS', 'BADNESS', 'DISSENT', 'INFORMALITY', 'ABSENCE', 'TRIBUNAL', 'BIOLOGY', 'EAGERNESS', 'WEALTH', 'FORMALITY', 'MONEY', 'SHARPNESS', 'STABILITY', 'EXPERIMENT', 'INVISIBILITY', 'ADVICE', 'NONCONFORMITY', 'INTUITION INSTINCT', 'SIMULTANEITY', 'FILAMENT', 'ACQUITTAL', 'RESTITUTION', 'ACCORD', 'ANACHRONISM', 'SPHERICITY ROTUNDITY', 'JEALOUSY', 'PRODUCTIVENESS', 'IRRESOLUTION', 'INTELLECTUAL', 'CONTRAPOSITION', 'UNIONISM LABOR UNION', 'CONTAINER', 'ELECTRONICS', 'SECLUSION', 'HONOR', 'MARKET', 'NONEXISTENCE', 'CONCISENESS', 'CLASSIFICATION', 'PROBABILITY', 'INTERVAL', 'MATERIALITY', 'TRACK AND FIELD', 'CONFINEMENT', 'LOCATION', 'SLOWNESS', 'APPORTIONMENT', 'MEDIOCRITY', 'SECURITY', 'FURROW', 'ILLUSION', 'NATIVENESS', 'PITY', 'HOSPITALITY WELCOME', 'CLEANNESS', 'HUMOROUSNESS', 'SOLEMNITY', 'TRANSIENCE', 'INGRATITUDE', 'SHOW BUSINESS THEATER', 'EXCRETION', 'PROTECTION', 'INORGANIC MATTER', 'RESONANCE', 'UNSAVORINESS', 'ARRANGEMENT', 'ADVERSITY', 'YOUTH', 'CARDPLAYING', 'GIVING', 'RESTRAINT', 'MORNING NOON', 'CLOTHING MATERIALS', 'CAREFULNESS', 'EVENT', 'IMPROBITY', 'PURSUIT', 'IMPULSE', 'AFFECTATION', 'ROCKETRY MISSILERY', 'REMAINDER', 'PROVISION EQUIPMENT', 'GENERALITY', 'PERSEVERANCE', 'SORCERY', 'DISILLUSIONMENT', 'PHILOSOPHY', 'TRAVELER', 'CHEAPNESS', 'REFRESHMENT', 'AUTOMATION', 'HAIR', 'MEASUREMENT OF TIME', 'EVENING NIGHT', 'REPULSION', 'LEGALITY', 'COOPERATION', 'PROMOTION', 'MARINER', 'SCULPTURE', 'REAR', 'STENCH', 'PLANTS', 'SOLILOQUY', 'REPEAL', 'INTRUSION', 'UNSUBSTANTIALITY', 'FINANCIAL CREDIT', 'SUBSTANTIALITY', 'RARITY', 'MUSICIAN', 'ATTRACTION', 'INTERIM', 'PERPETUITY', 'PULPINESS', 'ACCOMPANIMENT', 'DOUBLENESS', 'STOCK MARKET', 'FRIEND', 'INCOMPLETENESS', 'THEFT', 'WEIGHT', 'DENSITY', 'JUSTICE', 'PLAIN', 'ACTIVITY', 'HASTE', 'DEMOTION DEPOSAL', 'SKIING', 'CESSATION', 'INCURIOSITY', 'ARROGANCE', 'ORNAMENTATION', 'NONIMITATION', 'LOSS', 'INELEGANCE', 'REJOICING', 'COMMUNICATION', 'DISCORD', 'TOOLS MACHINERY', 'SECURITIES', 'WRONG', 'MISTEACHING', 'RADIO', 'INNOCENCE', 'SOPHISTRY', 'POWER POTENCY', 'DULLNESS', 'NONACCOMPLISHMENT', 'SADNESS', 'VIRTUE', 'SAVORINESS', 'DECEPTION', 'PUBLICATION', 'UNASTONISHMENT', 'DANGER', 'IMPORTANCE', 'EVOLUTION', 'TOP', 'HABITATION', 'DISAGREEMENT', 'ELASTICITY', 'TEXTURE', 'QUADRUPLICATION', 'THE PAST', 'SECRETION', 'MUSICAL INSTRUMENTS', 'CORRESPONDENCE', 'JUDGMENT', 'DESIRE', 'TREATISE', 'LIBRARY', 'ENCLOSURE', 'VULGARITY', 'VIOLENCE', 'SOCIAL CONVENTION', 'UNITED NATIONS INTERNATIONAL ORGANIZATIONS', 'PSYCHOLOGY PSYCHOTHERAPY', 'DISPLACEMENT', 'INSERTION', 'MANIFESTATION', 'WONDER', 'NUMEROUSNESS', 'ABNORMALITY', 'INFINITY', 'KINDNESS BENEVOLENCE', 'FORM', 'UNDERTAKING', 'WISE SAYING', 'COSTLESSNESS', 'RAIN', 'WAKEFULNESS', 'BANE', 'INTENTION', 'HEAT', 'MUSIC', 'ROTATION', 'LIQUIDITY', 'SAMENESS', 'DEFECTIVE VISION', 'PRODUCT', 'MISREPRESENTATION', 'BOASTING', 'ASSOCIATE', 'HARDNESS RIGIDITY', 'SOFTNESS PLIANCY', 'CONCURRENCE', 'RADAR RADIOLOCATORS', 'HEALTH', 'NEWS', 'LIBERATION', 'COMPLEXITY', 'LAMENTATION', 'THE FUTURE', 'DISCRIMINATION', 'PRETEXT', 'SKILL', 'REDNESS', 'ILLICIT BUSINESS', 'UNBELIEF', 'YOUNGSTER', 'SUBJECTION', 'FASTIDIOUSNESS', 'BLEMISH', 'UNACCUSTOMEDNESS', 'DEAFNESS', 'CERTAINTY', 'NORMALITY', 'DIFFICULTY', 'CONTENTION', 'THE BODY', 'PARALLELISM', 'TRUTH', 'CONTRARIETY', 'INEXPECTATION', 'ARCHITECTURE DESIGN', 'PUSHING THROWING', 'PREVIOUSNESS', 'FEELING', 'EQUALITY', 'TOUGHNESS', 'BOTTOM', 'PERMANENCE', 'INFLUENCE', 'SENSATION', 'OVERESTIMATION', 'FATIGUE', 'HOCKEY', 'FRONT', 'EARLINESS', 'FACILITY', 'RASHNESS', 'CONFORMITY', 'FREQUENCY', 'EXERTION', 'DEVIATION', 'PITILESSNESS', 'EVIDENCE PROOF', 'DISOBEDIENCE', 'PRECURSOR', 'ENERGY', 'LITTLENESS', 'PRESENCE', 'IDEA', 'DIRECTOR', 'SIGNS INDICATORS', 'BLACKNESS', 'LEARNING', 'IMPROVEMENT', 'CAUSE', 'PLAIN SPEECH', 'ARISTOCRACY NOBILITY GENTRY', 'UNSOCIABILITY', 'PRODIGALITY', 'ILLEGALITY', 'IMPATIENCE', 'COMBINATION', 'ADDITION', 'AIR WEATHER', 'FORMLESSNESS', 'CELIBACY', 'SOCIABILITY', 'TOPIC', 'BRIBERY', 'MATERIALS', 'TRANSPARENCY', 'SECRECY', 'LOUDNESS', 'EMERGENCE', 'PARSIMONY', 'CONTRACTION', 'OBSERVANCE', 'REGRET', 'SYMMETRY', 'CIRCUMSTANCE', 'BELIEF', 'COUNTRY', 'THRIFT', 'LEVERAGE PURCHASE', 'RETALIATION', 'FOUR', 'STRUCTURE', 'PERIODICAL', 'BOWLING', 'FOLLOWING', 'COLD', 'OSTENTATION', 'HATE', 'CONTEMPT', 'DISCOVERY', 'WILL', 'ILL HUMOR', 'PRESERVATION', 'CONVEXITY PROTUBERANCE', 'ROUTE PATH', 'APPEARANCE', 'PROSE', 'UNTIMELINESS', 'KILLING', 'INSUFFICIENCY', 'COURAGE', 'FEAR FEARFULNESS', 'CHASTITY', 'MEASUREMENT', 'CONVERGENCE', 'MOTIVATION INDUCEMENT', 'AGITATION', 'UNCHASTITY', 'UNREGRETFULNESS', 'PLUNGE', 'END', 'PIETY', 'PERMISSION', 'SANITY', 'ODOR', 'ASSENT', 'OBEDIENCE', 'ROUGHNESS', 'PREPARATION', 'NEUTRALITY', 'SHADE', 'WORKER DOER', 'UNDERESTIMATION', 'INSOLENCE', 'PROPERTY', 'AGRICULTURE', 'DEPUTY AGENT', 'DIFFUSENESS', 'AGREEMENT', 'ELEGANCE', 'DISPERSION', 'LEAP', 'VAPOR GAS', 'TIME', 'POSTERITY', 'TRANSFER OF PROPERTY OR RIGHT', 'CELEBRATION', 'TEDIUM', 'DISCONTENT', 'CURSE', 'CONGRATULATION', 'SOURNESS', 'UNPREPAREDNESS', 'PROBITY', 'SUBSTITUTION', 'OBSTINACY', 'IGNORANCE', 'FOLD', 'ONENESS', 'DEATH', 'BASEBALL', 'AMBIGUITY', 'SUFFICIENCY', 'POLITICS', 'ARTLESSNESS', 'SENSUALITY', 'NEGLECT', 'MECHANICS', 'LAND', 'WORD', 'EXAGGERATION', 'REVERSION', 'TIMELINESS', 'REFUSAL', 'INFERIORITY', 'BAD PERSON', 'SUBSTANCE ABUSE', 'INCREASE', 'LEFT SIDE', 'CRITICISM OF THE ARTS', 'TENDENCY', 'FITNESS EXERCISE', 'MASTER', 'UNCLEANNESS', 'INSIGNIFICANCE', 'BANTER', 'ASSOCIATION', 'QUANTITY', 'CRY CALL', 'REVOLUTION', 'HUMANKIND', 'ABODE HABITAT', 'SUCCESS', 'RESENTMENT ANGER', 'DEPARTURE', 'ADJUNCT', 'CONTENTMENT', 'INSTANTANEOUSNESS', 'LETTER', 'DIFFERENCE', 'REGRESSION', 'CHANNEL', 'CAPRICE', 'CIRCULARITY', 'IMPOSITION', 'ENMITY', 'BEGINNING', 'LENDING', 'REPRESENTATION DESCRIPTION', 'LIST', 'BUSINESSMAN MERCHANT', 'PROPHETS RELIGIOUS FOUNDERS', 'EXCITEMENT', 'DISARRANGEMENT', 'MARRIAGE', 'PREMONITION', 'MOISTURE', 'PLEASURE', 'MISANTHROPY', 'SPACE TRAVEL', 'RELATIONSHIP BY MARRIAGE', 'FOOL', 'ATTENTION', 'INTERPRETATION', 'DEGREE', 'UNGRAMMATICALNESS', 'SEWING', 'ENTRANCE', 'DEPTH', 'APPROVAL', 'DURATION', 'TASTE', 'ASCENT', 'IDOLATRY', 'ACQUISITION', 'CERAMICS', 'ANIMAL SOUNDS', 'ORANGENESS', 'BASKETBALL', 'COMMISSION', 'EJECTION', 'IMMATERIALITY', 'CONVERSATION', 'DEFENSE', 'FUEL', 'TEACHING', 'CLOUD', 'THE LAITY', 'CONSUMPTION', 'LAXNESS', 'JURISDICTION', 'UNSELFISHNESS', 'HEIGHT', 'RECESSION', 'UNPRODUCTIVENESS', 'RELINQUISHMENT', 'QUALIFICATION', 'NEGATION DENIAL', 'BODILY DEVELOPMENT', 'MATHEMATICS', 'UNIMPORTANCE', 'COPY', 'MEANINGLESSNESS', 'COMBATANT', 'OCCULTISM', 'RIGHT', 'VICE', 'DISORDER', 'DISPARAGEMENT', 'ANGULARITY', 'MEANING', 'SLEEP', 'DISAPPOINTMENT', 'FINANCE INVESTMENT', 'GRAYNESS', 'NONUNIFORMITY', 'IMPROBABILITY', 'EXPECTATION', 'POSSESSION', 'EXPANSION GROWTH', 'HUMILITY', 'NEARNESS', 'GUILT', 'SEPARATION', 'SEQUEL', 'INHOSPITALITY', 'DISTORTION', 'THOUGHT', 'HOPELESSNESS', 'UNHEALTHFULNESS', 'LAWYER', 'INVOLVEMENT', 'MINERALS METALS', 'DEFEAT', 'PRIDE', 'DISINTEGRATION', 'RELIGIOUS BUILDINGS', 'DEPRESSION', 'SEX', 'INSTRUMENTS OF PUNISHMENT', 'WILLINGNESS', 'PHOTOGRAPHY', 'TENNIS', 'BLUSTER', 'IMPERFECTION', 'UNINTELLIGENCE', 'ODORLESSNESS', 'EXTRACTION', 'COMPENSATION', 'WARNING', 'SHORTCOMING', 'CURIOSITY', 'CIRCUMSCRIPTION', 'MENTAL ATTITUDE', 'UNINTELLIGIBILITY', 'SHIP BOAT', 'RELIEF', 'INSIGNIA', 'TRISECTION', 'COLORLESSNESS', 'INEXPEDIENCE', 'NARROW MINDEDNESS', 'WEAKNESS', 'SEA OCEAN', 'COOKING', 'SERVILITY', 'TALKATIVENESS', 'SUBSEQUENCE', 'REFRIGERATION', 'GAMBLING', 'SATIETY', 'DEBT', 'GOOD PERSON', 'FALSENESS', 'INACTIVITY', 'SCHOOL', 'USELESSNESS', 'OPAQUENESS', 'VERTICALNESS', 'REGION', 'ATONEMENT', 'INSANITY MANIA', 'ORTHODOXY', 'COHESION', 'UNSANCTITY', 'BLUENESS', 'THE CLERGY', 'INTELLECT', 'REJECTION', 'FASTING', 'NUCLEAR PHYSICS', 'EXPENSIVENESS', 'INCREDULITY', 'TOBACCO', 'LIGHT', 'NONCOHESION', 'HORSE RACING', 'IMPERFECT SPEECH', 'DISAPPROVAL', 'DECREASE', 'EXTERIORITY', 'SPECTATOR', 'EXCESS', 'LATENESS', 'HELL', 'RESTORATION', 'CENTRALITY', 'COMPOSITION', 'PARTICIPATION', 'ABRIDGMENT', 'TOWN CITY', 'MODESTY', 'POLITICIAN', 'ROOM', 'OSCILLATION', 'INCLUSION', 'CLOTHING', 'INEXCITABILITY', 'BOXING', 'EVIL SPIRITS', 'POSSIBILITY', 'ORDER', 'LIABILITY', 'INTRINSICALITY', 'PLURALITY', 'ASCETICISM', 'OVERRUNNING', 'APPROACH', 'DRYNESS', 'MERCHANDISE', 'ARENA', 'LAKE POOL', 'SHALLOWNESS', 'FOOD', 'UNSKILLFULNESS', 'ELOQUENCE', 'MODERATION', 'STRENGTH', 'COMMAND', 'HEALTHFULNESS', 'THE COUNTRY', 'WHOLE', 'ACCUSATION', 'ELECTRICITY MAGNETISM', 'PATIENCE', 'TAKING', 'OFFER', 'DUENESS', 'AGGRAVATION', 'CURVATURE', 'UNNERVOUSNESS', 'SPELL CHARM', 'INQUIRY', 'CONDOLENCE', 'FURNITURE', 'TEACHER', 'WIND', 'GRATITUDE', 'STREAM', 'TITLE', 'VEHICLE', 'CLOSURE', 'OPTICAL INSTRUMENTS', 'PHRASE', 'DUPLICATION', 'PHYSICS', 'DISLIKE', 'REPETITION', 'VANITY', 'MIDDLE', 'FASHION', 'ROCK', 'UNIFORMITY', 'FORGIVENESS', 'ANGEL SAINT', 'THEORY SUPPOSITION', 'ARMS', 'AFFIRMATION', 'POVERTY', 'SUBTRACTION', 'FREEDOM', 'INTELLIGENCE WISDOM', 'PUNISHMENT', 'FIVE AND OVER', 'CUNNING', 'CONTINUITY', 'SIMPLICITY', 'YELLOWNESS', 'RELATIONSHIP BY BLOOD', 'THE MINISTRY', 'INLET GULF', 'COMFORT', 'CROSSING', 'FOOLISHNESS', 'DISRESPECT', 'ANSWER', 'STATE', 'MASCULINITY', 'SIZE LARGENESS', 'REPUTE', 'MISUSE', 'MANNER MEANS', 'ENDEAVOR', 'PLEASANTNESS', 'SEQUENCE', 'GRANDILOQUENCE', 'CONVERSION', 'SOCCER', 'MISBEHAVIOR', 'TRAVEL', 'GOVERNMENT', 'RIGHT SIDE', 'PREDETERMINATION', 'COMMUNICATIONS', 'ANIMAL HUSBANDRY', 'STRAIGHTNESS', 'IMITATION', 'STUDENT', 'NARROWNESS THINNESS', 'UGLINESS', 'INDIFFERENCE', 'SIDE', 'MEDIATION', 'CONTINUANCE', 'UNPLEASURE', 'ECCLESIASTICAL ATTIRE', 'DISCONTINUITY', 'REGULARITY OF RECURRENCE', 'THE UNIVERSE ASTRONOMY', 'DIVERGENCE', 'SEMILIQUIDITY', 'AIRCRAFT', 'CONCAVITY', 'EXPLOSIVE NOISE', 'PROHIBITION', 'GRAPHIC ARTS', 'LEGAL ACTION', 'AID', 'ETHICS', 'FLATTERY', 'CHEERFULNESS', 'POWDERINESS CRUMBLINESS', 'INDECENCY', 'IRREGULARITY OF RECURRENCE', 'SUBMISSION', 'FEWNESS', 'CHANGING OF MIND', 'INATTENTION', 'ACCOMPLISHMENT', 'ADULT OR OLD PERSON', 'REVENGE', 'FORGETFULNESS', 'RECORD', 'SPEECH', 'CHANGE', 'COUNCIL', 'CIRCUITOUSNESS', 'ANONYMITY', 'RECORDER', 'RELIGIOUS RITES', 'PLAINNESS', 'SPECTER', 'MEAN', 'ABSENCE OF INFLUENCE', 'BEAUTY', 'REACTION', 'LEADING', 'LEGISLATURE GOVERNMENT ORGANIZATION', 'FOOTBALL', 'EFFECT', 'SOCIAL CLASS AND STATUS', 'ANCESTRY', 'TRANSFERAL TRANSPORTATION', 'HORIZONTALNESS', 'BODY OF LAND', 'HARMONICS MUSICAL ELEMENTS', 'PRICE FEE', 'OCCUPATION', 'WORKPLACE', 'COVERING', 'LATENT MEANINGFULNESS', 'LIGHTNESS', 'ANIMALS INSECTS', 'PACIFICATION', 'POSSESSOR', 'SPORTS', 'BREADTH THICKNESS', 'DISUSE', 'PROMISE', 'ATTACK', 'EXTRINSICALITY', 'MEMORY', 'EXISTENCE', 'MISJUDGMENT', 'INACTION', 'BISECTION', 'SENSATIONS OF TOUCH', 'UNCOMMUNICATIVENESS', 'ESCAPE', 'THREE', 'DISCOUNT', 'PART', 'PREJUDGMENT', 'REPRODUCTION PROCREATION', 'SMOOTHNESS', 'EARTH SCIENCE', 'GLUTTONY', 'RECEIVING', 'ELEVATION', 'OPENING', 'DISSUASION', 'CONSENT', 'DARKNESS DIMNESS', 'IMPULSE IMPACT', 'SPECIALTY', 'VICTORY', 'RESINS GUMS', 'NONOBSERVANCE', 'MULTIFORMITY', 'SHAFT', 'INVERSION', 'STRICTNESS', 'PENDENCY', 'BENEFACTOR', 'REQUEST', 'FICTION', 'PENALTY', 'SUPERIORITY', 'ABSENCE OF THOUGHT', 'RECEPTION', 'REST REPOSE', 'LAWLESSNESS', 'WRITING', 'ENVY', 'IMPOTENCE', 'COWARDICE', 'PURCHASE', 'TRIPLICATION', 'INFREQUENCY', 'DISAPPEARANCE', 'RESOLUTION', 'DISTANCE REMOTENESS', 'NOMENCLATURE', 'AUTHORITY', 'REASONING', 'LENGTH', 'PARTICULARITY', 'BIRTH', 'EXPEDIENCE', 'PAYMENT', 'SOUND', 'UNKINDNESS MALEVOLENCE', 'PAIN', 'DESCENT', 'PRECEDENCE', 'CUSTOM HABIT', 'LIQUEFACTION', 'ALARM', 'DUPE', 'HEALTH CARE', 'MESSENGER', 'PERFECTION', 'UNIMAGINATIVENESS', 'HEARING', 'NUTRITION', 'HEAVEN', 'VARIEGATION', 'LANGUAGE', 'RESPECT', 'COLOR', 'OILS LUBRICANTS', 'SPELL', 'POETRY', 'LOVE', 'THE ENVIRONMENT', 'MOTION', 'TIMELESSNESS', 'DISREPUTE', 'INEQUALITY', 'PULLING', 'RIDICULE', 'NECESSITY', 'OLDNESS', 'BUBBLE', 'VISION', 'NEWNESS', 'INFORMATION', 'CONDEMNATION', 'LOWNESS', 'THREAT', 'DEITY', 'NONPAYMENT', 'CHEMISTRY CHEMICALS', 'HEATING', 'CHANGEABLENESS', 'COURTESY', 'GOODNESS', 'DEFIANCE', 'ENTERTAINER', 'LENIENCY', 'SWIFTNESS', 'REFUGE', 'RECEIPTS', 'FORESIGHT', 'MIXTURE', 'SANCTITY', 'BOUNDS', 'CAUTION', 'WORSHIP', 'RELATION', 'ALLUREMENT', 'DIRECTION MANAGEMENT', 'HOPE', 'OPPONENT', 'COMMERCE ECONOMICS', 'EATING', 'SOBRIETY', 'UNDUENESS', 'CORRELATION', 'DISCOURTESY', 'FRICTION', 'FEMININITY', 'COMPUTER SCIENCE', 'INHABITANT NATIVE', 'MODEL', 'AVOIDANCE', 'DICTION', 'MOTION PICTURES', 'INTEMPERANCE', 'ATTRIBUTION', 'LOVEMAKING ENDEARMENT', 'MISINTERPRETATION', 'SCRIPTURE', 'SIMILARITY', 'ABANDONMENT', 'SALE', 'IMMINENCE', 'FIGURE OF SPEECH', 'EXTRANEOUSNESS', 'BROWNNESS', 'TOUCH', 'IMPAIRMENT', 'SUPPORT', 'OBLIQUITY', 'PURPLENESS', 'SIBILATION', 'GRAMMAR', 'RELAPSE', 'PROGRESSION', 'FAINTNESS OF SOUND', 'TELEVISION', 'CHOICE', 'MARSH', 'INJUSTICE', 'LACK OF FEELING', 'ORGANIC MATTER', 'LIGHT SOURCE', 'BRITTLENESS FRAGILITY', 'NONRELIGIOUSNESS', 'SEMITRANSPARENCY', 'INSENSIBILITY', 'SOLUTION', 'ANALYSIS', 'FAILURE', 'BOOK', 'SAFETY', 'PUNGENCY', 'DECEIVER', 'BLINDNESS', 'SHORTNESS', 'COMPULSION', 'WISE PERSON', 'JUDGE JURY', 'PRODUCTION', 'RETENTION', 'DUTY', 'INTERCHANGE', 'USE', 'ANXIETY', 'GOLF', 'EXCLUSION', 'WATER TRAVEL', 'DISACCORD', 'GREATNESS', 'RESISTANCE', 'ECCENTRICITY', 'LITERATURE', 'OPPOSITION', 'AVIATION', 'DISCLOSURE', 'TEMPERANCE', 'AUTOMOBILE RACING', 'SWEETNESS', 'CONTENTS', 'WARFARE', 'ARRIVAL', 'COMPARISON', 'JUSTIFICATION', 'CREDULITY', 'EXPENDITURE', 'INTERPOSITION', 'IMPIETY', 'THE PRESENT', 'RADIATION RADIOACTIVITY', 'THIEF', 'THEOLOGY', 'PRECEPT', 'EVILDOER', 'QUADRISECTION', 'COMPROMISE', 'WRONGDOING', 'INTOXICATION ALCOHOLIC DRINK', 'OPERATION', 'JOINING', 'CONCEALMENT', 'AVIATOR', 'ASSEMBLAGE', 'DESTRUCTION', 'INSIPIDNESS', 'HIGHLANDS', 'SPACE', 'SEASON', 'PREROGATIVE', 'DISEASE', 'LIBERALITY', 'UNORTHODOXY', 'DISTRACTION CONFUSION', 'REMEDY', 'PLAN', 'CHANCE', 'HINDRANCE', 'WHITENESS', 'SERVANT EMPLOYEE', 'HISTORY', 'NERVOUSNESS', 'VISUAL ARTS', 'LAYER', 'INTELLIGIBILITY', 'DISSIMILARITY', 'IMPOSSIBILITY', 'ERROR', 'BLUNTNESS', 'IMAGINATION', 'INTERMENT', 'SELFISHNESS', 'TASTE TASTEFULNESS', 'THERAPY MEDICAL TREATMENT', 'REPEATED SOUNDS', 'AMUSEMENT', 'SANCTIMONY']

DEFAULT_TIMEZONE = pytz.utc
START_TIME = datetime.datetime(year=2000, month=1, day=1)

SPTOK_NU = '<NU>'
SPTOK_ME = '<ME>'
SPTOK_OTHER = '<OTHER>'
SPTOK_UNK = '<UNK>'
SPTOK_NL = '<NL>'
SPTOK_EOS = '<EOS>'
SPTOK_SOS = '<SOS>'

SPTOK_REPS = ['URL', 'REDDIT_USERNAME', 'SUBREDDIT_NAME']

def map_pos(inpos):
    outpos = inpos
    if inpos in ['NNP', 'NNS', 'NNPS', 'NN']:
        outpos = 'NN'
    elif inpos in ['JJS', 'JJR', 'JJ']:
        outpos = 'JJ'
    elif inpos in ['RBR', 'RBS', 'RB']:
        outpos = 'RB'
    elif inpos in ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VB']:
        outpos = 'VB'
    elif inpos in ['#', '$', '\'\'', ',', '.', ':']:
        outpos = 'PUNCT'
    elif inpos in ['PRP', 'PRP$', 'WP$', 'WP']:
        outpos = 'PR'
    elif inpos in ['PDT', 'DT', 'WDT']:
        outpos = 'DT'
    elif inpos in ['FW']:
        outpos = 'FW'
    elif inpos in ['IN']:
        outpos = 'IN'
    else:
        outpos = 'OTHER'
    return outpos

POS_LSET = ['RBR', 'WDT', 'LS', 'VBZ', 'RBS', 'CD', 'JJ', '``', 'POS', 'NN', '#', 'RP', 'MD', 'RB', 'VBG', 'FW', 'WP', 'JJR', 'PRP', 'PRP$', 'DT', 'PDT', 'VB', 'WRB', 'VBN', ',', 'UH', 'SYM', 'JJS', ':', 'VBD', 'NNS', '.', 'TO', 'NNP', 'UNK', 'CC', 'EX', "''", 'NNPS', 'VBP', 'WP$', 'IN', '$']
POS_LOOKUP = {}
with open(dir_path + '/pos_dist_out_above_95') as handle:
    for line in handle:
        parts = line.strip().split(' --- ')
        POS_LOOKUP[parts[0]] = parts[1] # map_pos(parts[1])
POS_DSET = ['NN', 'JJ', 'RB', 'VB', 'PUNCT', 'PR', 'DT', 'FW', 'IN', 'OTHER']
POS_LSET.sort()
POS_DSET.sort()

def get_post_id(idstr):
    num_id = None
    pid_parts = idstr.split('_')
    if len(pid_parts) == 1:
        #print(idstr)
        num_id = base36.loads(pid_parts[0])
    elif len(pid_parts) == 2:
        # print(pid_parts[1])
        num_id = base36.loads(pid_parts[1])
    else:
        print('Error: The parent ID \'' + idstr + '\' is not valid.')
        sys.exit(1)
    return num_id

def text_clean(instr):
    outstr = instr.strip().lower()
    outstr = outstr.replace('\n', ' ')
    outstr = replace_urls(outstr)
    # remove html chars
    outstr = html.unescape(outstr)
    # general replace
    outstr = re.sub('[^a-z0-9 ]', '', outstr)
    return outstr

def open_for_write(filename, binary=False):
    mkpath = '/'.join(filename.split('/')[:-1])
    os.makedirs(mkpath, exist_ok=True)
    return open(filename, 'w' + ('b' if binary else ''))

def replace_urls(instr):
    # replace reddit tags
    outstr = re.sub('\[URL=.*?\\\\?URL\]', 'URL', instr)
    # outstr = re.sub('\[\\?URL=.*?\]', 'URL', instr)
    # replace general urls
    outstr = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', outstr)
    # replace reddit usernames and subreddits
    outstr = re.sub('/?u/[A-Za-z0-9\-_]+', 'REDDIT_USERNAME', outstr)
    outstr = re.sub('/?r/[A-Za-z0-9\-_]+', 'SUBREDDIT_NAME', outstr)
    return outstr

def gprint(msg, logname, error=False, important=False, ptime=True, p2c=True):
    tmsg = msg if not important else colored(msg, 'cyan')
    tmsg = tmsg if not error else colored(msg, 'red')
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    cmsg = str(st) + ': ' + str(tmsg) if ptime else str(tmsg)
    tmsg = str(st) + ': ' + str(msg) if ptime else str(msg)
    if p2c:
        print(cmsg)
    log_file = open(dir_path + '/logs/' + logname + '.log', 'a')
    log_file.write(tmsg + '\n')
    log_file.flush()
    log_file.close()
