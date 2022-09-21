TAG_SCHEME_COMBINED = [
    'B-loc',
    'B-misc',
    'B-org',
    'B-per',
    'B-pro',
    'B-evt',
    'B-deriv-per',
    'I-loc',
    'I-misc',
    'I-org',
    'I-per',
    'I-pro',
    'I-evt',
    'I-deriv-per',
    'O'
]

TAG_SCHEME_SSK = [
    'B-loc',
    'B-misc',
    'B-org',
    'B-per',
    'B-deriv-per',
    'I-loc',
    'I-misc',
    'I-org',
    'I-per',
    'I-deriv-per',
    'O'
]

TAG_SCHEME_BSNLP = [
    'B-loc',
    'B-misc',
    'B-org',
    'B-per',
    'B-pro',
    'B-evt',
    'I-loc',
    'I-misc',
    'I-org',
    'I-per',
    'I-pro',
    'I-evt',
    'O'
]

TRAINING_HYPERPARAMETERS = {
    'epochs' : 15,
    'warmup_steps' : 100,
    'train_batch_size': 13,
    'learning_rate': 3e-5
}

COUNTRIES = ['afganistan', 'albanija', 'alžirija', 'andora', 'angola', 'antigva in barbuda', 'argentina', 'armenija', 'avstralija', 'avstrija', 'azerbajdžan', 'bahami', 'bahrajn', 'bangladeš', 'barbados', 'belgija', 'belize', 'belorusija', 'benin', 'bocvana', 'bolgarija', 'bolivija', 'bosna in hercegovina', 'brazilija', 'brunej', 'burkina faso', 'burundi', 'butan', 'ciper', 'čad', 'češka', 'čile', 'črna gora', 'danska', 'dominika', 'dominikanska republika', 'džibuti', 'egipt', 'ekvador', 'ekvatorialna gvineja', 'eritreja', 'estonija', 'esvatini', 'etiopija', 'fidži', 'filipini', 'finska', 'francija', 'gabon', 'gambija', 'gana', 'grčija', 'grenada', 'gruzija', 'gvajana', 'gvatemala', 'gvineja', 'gvineja bissau', 'haiti', 'honduras', 'hrvaška', 'indija', 'indonezija', 'irak', 'iran', 'irska', 'islandija', 'italija', 'izrael', 'jamajka', 'japonska', 'jemen', 'jordanija', 'južna afrika', 'južna koreja', 'južni sudan', 'kambodža', 'kamerun', 'kanada', 'katar', 'kazahstan', 'kenija', 'kirgizistan', 'kiribati', 'ljudska republika kitajska', 'kolumbija', 'komori', 'kongo', 'demokratična republika kongo', 'kostarika', 'kuba', 'kuvajt', 'laos', 'latvija', 'lesoto', 'libanon', 'liberija', 'libija', 'lihtenštajn', 'litva', 'luksemburg', 'madagaskar', 'madžarska', 'malavi', 'maldivi', 'malezija', 'mali', 'malta', 'maroko', 'marshallovi otoki', 'mavricij', 'mavretanija', 'mehika', 'mikronezija', 'mjanmar', 'moldavija', 'monako', 'mongolija', 'mozambik', 'namibija', 'nauru', 'nemčija', 'nepal', 'niger', 'nigerija', 'nikaragva', 'nizozemska', 'norveška', 'nova zelandija', 'oman', 'pakistan', 'palav', 'palestina', 'panama', 'papua nova gvineja', 'paragvaj', 'peru', 'poljska', 'portugalska', 'romunija', 'ruanda', 'rusija', 'salomonovi otoki', 'salvador', 'samoa', 'san marino', 'saudova arabija', 'sejšeli', 'senegal', 'severna koreja', 'severna makedonija', 'sierra leone', 'singapur', 'sirija', 'slonokoščena obala', 'slovaška', 'slovenija', 'somalija', 'srbija', 'srednjeafriška republika', 'sudan', 'surinam', 'sveta lucija', 'sveti krištof in nevis', 'sveti tomaž in princ', 'sveti vincencij in grenadine', 'španija', 'šrilanka', 'švedska', 'švica', 'tadžikistan', 'tajska', 'tanzanija', 'togo', 'tonga', 'trinidad in tobago', 'tunizija', 'turčija', 'turkmenistan', 'tuvalu', 'uganda', 'ukrajina', 'urugvaj', 'uzbekistan', 'vanuatu', 'vatikan', 'združeno kraljestvo', 'venezuela', 'vietnam', 'vzhodni timor', 'zambija', 'zda', 'združeni arabski emirati', 'zelenortski otoki', 'zimbabve', 'abhazija\narcah', 'cookovi otoki', 'kosovo', 'niue', 'severni ciper', 'zahodna sahara', 'somaliland', 'južna osetija', 'tajvan', 'pridnestrje']
SLOVENIAN_CITIES = ['ajdovščina', 'bled', 'bovec', 'brežice', 'celje', 'cerknica', 'črnomelj', 'domžale', 'dravograd', 'gornja radgona', 'grosuplje', 'hrastnik', 'idrija', 'ilirska bistrica', 'izola', 'jesenice', 'kamnik', 'kobarid', 'kočevje', 'koper', 'kostanjevica na krki', 'kranj', 'krško', 'laško', 'lenart v slovenskih goricah', 'lendava', 'litija', 'ljubljana', 'ljutomer', 'logatec', 'maribor', 'medvode', 'mengeš', 'metlika', 'mežica', 'murska sobota', 'nova gorica', 'novo mesto', 'ormož', 'piran', 'postojna', 'prevalje', 'ptuj', 'radeče', 'radovljica', 'ravne na koroškem', 'ribnica', 'rogaška slatina', 'ruše', 'sevnica', 'sežana', 'slovenj gradec', 'slovenska bistrica', 'slovenske konjice', 'šempeter pri gorici', 'šentjur', 'škofja loka', 'šoštanj', 'tolmin', 'trbovlje', 'trebnje', 'tržič', 'turnišče', 'velenje', 'vipava', 'vipavski križ', 'višnja gora', 'vrhnika', 'zagorje ob savi', 'zreče', 'žalec', 'železniki', 'žiri']
SLOVENIAN_MUNICIPALITIES = ['ajdovščina', 'ankaran', 'apače', 'beltinci', 'benedikt', 'bistrica ob sotli', 'bled', 'bloke', 'bohinj', 'borovnica', 'bovec', 'braslovče', 'brda', 'brezovica', 'brežice', 'cankova', 'celje', 'cerklje na gorenjskem', 'cerknica', 'cerkno', 'cerkvenjak', 'cirkulane', 'črenšovci', 'črna na koroškem', 'črnomelj', 'destrnik', 'divača', 'dobje', 'dobrepolje', 'dobrna', 'dobrova - polhov gradec', 'dobrovnik', 'dol pri ljubljani', 'dolenjske toplice', 'domžale', 'dornava', 'dravograd', 'duplek', 'gorenja vas - poljane', 'gorišnica', 'gorje', 'gornja radgona', 'gornji grad', 'gornji petrovci', 'grad', 'grosuplje', 'hajdina', 'hoče - slivnica', 'hodoš', 'horjul', 'hrastnik', 'hrpelje - kozina', 'idrija', 'ig', 'ilirska bistrica', 'ivančna gorica', 'izola', 'jesenice', 'jezersko', 'juršinci', 'kamnik', 'kanal ob soči', 'kidričevo', 'kobarid', 'kobilje', 'kočevje', 'komen', 'komenda', 'koper', 'kostanjevica na krki', 'kostel', 'kozje', 'kranj', 'kranjska gora', 'križevci', 'krško', 'kungota', 'kuzma', 'laško', 'lenart', 'lendava', 'litija', 'ljubljana', 'ljubno', 'ljutomer', 'log - dragomer', 'logatec', 'loška dolina', 'loški potok', 'lovrenc na pohorju', 'luče', 'lukovica', 'majšperk', 'makole', 'maribor', 'markovci', 'medvode', 'mengeš', 'metlika', 'mežica', 'miklavž na dravskem polju', 'miren - kostanjevica', 'mirna', 'mirna peč', 'mislinja', 'mokronog - trebelno', 'moravče', 'moravske toplice', 'mozirje', 'murska sobota', 'muta', 'naklo', 'nazarje', 'nova gorica', 'novo mesto', 'odranci', 'oplotnica', 'ormož', 'osilnica', 'pesnica', 'piran', 'pivka', 'podčetrtek', 'podlehnik', 'podvelka', 'poljčane', 'polzela', 'postojna', 'prebold', 'preddvor', 'prevalje', 'ptuj', 'puconci', 'rače - fram', 'radeče', 'radenci', 'radlje ob dravi', 'radovljica', 'ravne na koroškem', 'razkrižje', 'rečica ob savinji', 'renče - vogrsko', 'ribnica', 'ribnica na pohorju', 'rogaška slatina', 'rogašovci', 'rogatec', 'ruše', 'selnica ob dravi', 'semič', 'sevnica', 'sežana', 'slovenj gradec', 'slovenska bistrica', 'slovenske konjice', 'sodražica', 'solčava', 'središče ob dravi', 'starše', 'straža', 'sveta ana', 'sveta trojica v slovenskih goricah', 'sveti andraž v slovenskih goricah', 'sveti jurij ob ščavnici', 'sveti jurij v slovenskih goricah', 'sveti tomaž', 'šalovci', 'šempeter - vrtojba', 'šenčur', 'šentilj', 'šentjernej', 'šentjur', 'šentrupert', 'škocjan', 'škofja loka', 'škofljica', 'šmarje pri jelšah', 'šmarješke toplice', 'šmartno pri litiji', 'šmartno ob paki', 'šoštanj', 'štore', 'tabor', 'tišina', 'tolmin', 'trbovlje', 'trebnje', 'trnovska vas', 'trzin', 'tržič', 'turnišče', 'velenje', 'velika polana', 'velike lašče', 'veržej', 'videm', 'vipava', 'vitanje', 'vodice', 'vojnik', 'vransko', 'vrhnika', 'vuzenica', 'zagorje ob savi', 'zavrč', 'zreče', 'žalec', 'železniki', 'žetale', 'žiri', 'žirovnica', "žužemberk"]
MONTHS = ["januarja", "februarja", "marec", "april", "maj", "junij", "julij", "avgust", "september", "oktober", "november", "december"]