"""Reference list in the style of Clinical Pharmacology & Therapeutics.

All authors when there are at most six, otherwise the first author followed by "et al."; journal names abbreviated
(Index Medicus, with periods) and italicized (between *asterisks*); volume, full page range or article number, year.
Records were checked against Crossref/PubMed on 2026-10-01 (paper/references_verified.md and the verification notes).
"""

REFERENCES = {
    'Sheiner1977': 'Sheiner, L.B., Rosenberg, B. & Marathe, V.V. Estimation of population characteristics of '
                   'pharmacokinetic parameters from routine clinical data. *J. Pharmacokinet. Biopharm.* 5, 445–479 '
                   '(1977).',
    'Mould2012': 'Mould, D.R. & Upton, R.N. Basic concepts in population modeling, simulation, and model-based drug '
                 'development. *CPT Pharmacometrics Syst. Pharmacol.* 1, e6 (2012).',
    'Marshall2016': 'Marshall, S.F. et al. Good practices in model-informed drug discovery and development: practice, '
                    'application, and documentation. *CPT Pharmacometrics Syst. Pharmacol.* 5, 93–122 (2016).',
    'Madabushi2022': 'Madabushi, R., Seo, P., Zhao, L., Tegenge, M. & Zhu, H. Review: role of model-informed drug '
                     'development approaches in the lifecycle of drug development and regulatory decision-making. '
                     '*Pharm. Res.* 39, 1669–1680 (2022).',
    'FDA2022': 'US Food and Drug Administration. Population pharmacokinetics: guidance for industry '
               '<https://www.fda.gov/regulatory-information/search-fda-guidance-documents/population-pharmacokinetics> '
               '(2022). Accessed 1 October 2026.',
    'Byon2013': 'Byon, W. et al. Establishing best practices and guidance in population modeling: an experience with '
                'an internal population pharmacokinetic analysis guidance. *CPT Pharmacometrics Syst. Pharmacol.* 2, '
                'e51 (2013).',
    'Nguyen2017': 'Nguyen, T.H.T. et al. Model evaluation of continuous data pharmacometric models: metrics and '
                  'graphics. *CPT Pharmacometrics Syst. Pharmacol.* 6, 87–109 (2017).',
    'Jonsson1998': 'Jonsson, E.N. & Karlsson, M.O. Automated covariate model building within NONMEM. *Pharm. Res.* '
                   '15, 1463–1468 (1998).',
    'Lindbom2005': 'Lindbom, L., Pihlgren, P. & Jonsson, E.N. PsN-Toolkit—a collection of computer intensive '
                   'statistical methods for non-linear mixed effect modeling using NONMEM. *Comput. Methods Programs '
                   'Biomed.* 79, 241–257 (2005).',
    'Sale2015': 'Sale, M. & Sherer, E.A. A genetic algorithm based global search strategy for population '
                'pharmacokinetic/pharmacodynamic model selection. *Br. J. Clin. Pharmacol.* 79, 28–39 (2015).',
    'Ismail2022': 'Ismail, M. et al. Development of a genetic algorithm and NONMEM workbench for automating and '
                  'improving population pharmacokinetic/pharmacodynamic model selection. *J. Pharmacokinet. '
                  'Pharmacodyn.* 49, 243–256 (2022).',
    'Li2024': 'Li, X. et al. pyDarwin: a machine learning enhanced automated nonlinear mixed-effect model selection '
              'toolbox. *Clin. Pharmacol. Ther.* 115, 758–773 (2024).',
    'Chen2024': 'Chen, X. et al. A fully automatic tool for development of population pharmacokinetic models. *CPT '
                'Pharmacometrics Syst. Pharmacol.* 13, 1784–1797 (2024).',
    'Richardson2025': 'Richardson, S. et al. A machine learning approach to population pharmacokinetic modelling '
                      'automation. *Commun. Med. (Lond.)* 5, 327 (2025).',
    'Bram2026': 'Bräm, D.S., Steiert, B., Steffens, B., Pfister, M. & Koch, G. Automated pharmacometric model '
                'development by leveraging low-dimensional neural ODEs and LASSO regression. *CPT Pharmacometrics '
                'Syst. Pharmacol.* 15, e70285 (2026).',
    'McComb2022': 'McComb, M., Bies, R. & Ramanathan, M. Machine learning in pharmacometrics: opportunities and '
                  'challenges. *Br. J. Clin. Pharmacol.* 88, 1482–1499 (2022).',
    'Huang2024': 'Huang, Z., Denti, P., Mistry, H. & Kloprogge, F. Machine learning and artificial intelligence in '
                 'PK-PD modeling: fad, friend, or foe? *Clin. Pharmacol. Ther.* 115, 652–654 (2024).',
    'Yao2023': 'Yao, S. et al. ReAct: synergizing reasoning and acting in language models. In *Proceedings of the '
               '11th International Conference on Learning Representations* (ICLR, Kigali, 2023).',
    'Schick2023': 'Schick, T. et al. Toolformer: language models can teach themselves to use tools. In *Advances in '
                  'Neural Information Processing Systems 36* 68539–68551 (Curran Associates, Red Hook, 2023).',
    'Bran2024': 'Bran, A.M., Cox, S., Schilter, O., Baldassari, C., White, A.D. & Schwaller, P. Augmenting large '
                'language models with chemistry tools. *Nat. Mach. Intell.* 6, 525–535 (2024).',
    'Boiko2023': 'Boiko, D.A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language '
                 'models. *Nature* 624, 570–578 (2023).',
    'Cloesmeijer2024': 'Cloesmeijer, M.E., Janssen, A., Koopman, S.F., Cnossen, M.H. & Mathôt, R.A.A. ChatGPT in '
                       'pharmacometrics? Potential opportunities and limitations. *Br. J. Clin. Pharmacol.* 90, '
                       '360–365 (2024).',
    'Shin2024a': 'Shin, E. & Ramanathan, M. Evaluation of prompt engineering strategies for pharmacokinetic data '
                 'analysis with the ChatGPT large language model. *J. Pharmacokinet. Pharmacodyn.* 51, 101–108 (2024).',
    'Shin2024b': 'Shin, E., Yu, Y., Bies, R.R. & Ramanathan, M. Evaluation of ChatGPT and Gemini large language '
                 'models for pharmacometrics with NONMEM. *J. Pharmacokinet. Pharmacodyn.* 51, 187–197 (2024).',
    'Cha2025': 'Cha, H.J., Choe, K., Shin, E., Ramanathan, M. & Han, S. Leveraging large language models in '
               'pharmacometrics: evaluation of NONMEM output interpretation and simulation capabilities. *J. '
               'Pharmacokinet. Pharmacodyn.* 52, 34 (2025).',
    'Zheng2025': 'Zheng, W., Wang, W., Kirkpatrick, C.M.J., Landersdorfer, C.B., Yao, H. & Zhou, J. AI for NONMEM '
                 'coding in pharmacometrics research and education: shortcut or pitfall? *CPT Pharmacometrics Syst. '
                 'Pharmacol.* 14, 1965–1969 (2025).',
    'PritchardBell2026': 'Pritchard-Bell, A., Lin, C.W., Holmes, W. & Doshi, S. Context engineering for AI-assisted '
                         'pharmacometrics: a practical tutorial. *CPT Pharmacometrics Syst. Pharmacol.* 15, e70317 '
                         '(2026).',
    'Bloomingdale2026': 'Bloomingdale, P. & Khot, A. PMxAgent: an agentic platform for pharmacometrics. *CPT '
                        'Pharmacometrics Syst. Pharmacol.* 15, e70325 (2026).',
    'Saini2025': 'Saini, A. & Farnoud, A. QSP-Copilot: an AI-augmented platform for accelerating quantitative systems '
                 'pharmacology model development. *CPT Pharmacometrics Syst. Pharmacol.* 14, 1775–1786 (2025).',
    'Shahin2025': 'Shahin, M.H., Goswami, S., Lobentanzer, S. & Corrigan, B.W. Agents for change: artificial '
                  'intelligent workflows for quantitative clinical pharmacology and translational sciences. *Clin. '
                  'Transl. Sci.* 18, e70188 (2025).',
    'Kwack2026': 'Kwack, H., Kong, H., Lim, J., Zhang, B.-T., Hahn, J. & Chang, M.J. PKGPT: expert-orchestrated '
                 'recursive LLM agent for automated NONMEM PopPK modeling with human benchmarking. *Pharmaceutics* '
                 '18, 501 (2026).',
    'Terranova2024': 'Terranova, N. et al. Artificial intelligence for quantitative modeling in drug discovery and '
                     'development: an Innovation and Quality Consortium perspective on use cases and best practices. '
                     '*Clin. Pharmacol. Ther.* 115, 658–672 (2024).',
    'FDA2025AI': 'US Food and Drug Administration. Considerations for the use of artificial intelligence to support '
                 'regulatory decision-making for drug and biological products: draft guidance for industry and other '
                 'interested parties <https://www.fda.gov/regulatory-information/search-fda-guidance-documents/'
                 'considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological> '
                 '(2025). Accessed 1 October 2026.',
    'Lu2026': 'Lu, J. & Desikan, R. Quantitative systems pharmacology modeling amid the rise of agentic AI. *CPT '
              'Pharmacometrics Syst. Pharmacol.* 15, e70249 (2026).',
    'McCoy2026': 'McCoy, M. & McCoy, M. From executor to orchestrator: the pharmacology scientist in the age of '
                 'agentic AI. *Clin. Pharmacol. Ther.* 120, 648–662 (2026).',
    'Kong2025': 'Kong, H., Kim, I. & Zhang, B.-T. PKPy: a Python-based framework for automated population '
                'pharmacokinetic analysis. *PeerJ* 13, e20258 (2025).',
    'PKPy2': 'Kong, H. PKPy2: a Python framework for population pharmacokinetic and pharmacodynamic estimation. '
             'GitHub <https://github.com/Gumgo91/PKPy2> (2026). Accessed 1 October 2026.',
    'Savic2009': 'Savic, R.M. & Karlsson, M.O. Importance of shrinkage in empirical Bayes estimates for diagnostics: '
                 'problems and solutions. *AAPS J.* 11, 558–569 (2009).',
    'Hooker2007': 'Hooker, A.C., Staatz, C.E. & Karlsson, M.O. Conditional weighted residuals (CWRES): a model '
                  'diagnostic for the FOCE method. *Pharm. Res.* 24, 2187–2197 (2007).',
    'Comets2008': 'Comets, E., Brendel, K. & Mentré, F. Computing normalised prediction distribution errors to '
                  'evaluate nonlinear mixed-effect models: the npde add-on package for R. *Comput. Methods Programs '
                  'Biomed.* 90, 154–166 (2008).',
    'Bergstrand2011': 'Bergstrand, M., Hooker, A.C., Wallin, J.E. & Karlsson, M.O. Prediction-corrected visual '
                      'predictive checks for diagnosing nonlinear mixed-effects models. *AAPS J.* 13, 143–151 (2011).',
    'Wang2007': 'Wang, Y. Derivation of various NONMEM estimation methods. *J. Pharmacokinet. Pharmacodyn.* 34, '
                '575–593 (2007).',
    'Pinheiro1995': 'Pinheiro, J.C. & Bates, D.M. Approximations to the log-likelihood function in the nonlinear '
                    'mixed-effects model. *J. Comput. Graph. Stat.* 4, 12–35 (1995).',
    'Bauer2019': 'Bauer, R.J. NONMEM tutorial part II: estimation methods and advanced examples. *CPT Pharmacometrics '
                 'Syst. Pharmacol.* 8, 538–556 (2019).',
    'Byrd1995': 'Byrd, R.H., Lu, P., Nocedal, J. & Zhu, C. A limited memory algorithm for bound constrained '
                'optimization. *SIAM J. Sci. Comput.* 16, 1190–1208 (1995).',
    'Grasela1985': 'Grasela, T.H., Jr. & Donn, S.M. Neonatal population pharmacokinetics of phenobarbital derived from '
                   'routine clinical data. *Dev. Pharmacol. Ther.* 8, 374–383 (1985).',
    'nlmixr2data': 'Fidler, M. & Wang, W. nlmixr2data: nonlinear mixed effects models in population PK/PD, data. R '
                   'package version 2.0.10 <https://CRAN.R-project.org/package=nlmixr2data> (2026). Accessed 1 '
                   'October 2026.',
    'Minto1997': 'Minto, C.F. et al. Influence of age and gender on the pharmacokinetics and pharmacodynamics of '
                 'remifentanil. I. Model development. *Anesthesiology* 86, 10–23 (1997).',
    'Schoemaker2019': 'Schoemaker, R. et al. Performance of the SAEM and FOCEI algorithms in the open-source, '
                      'nonlinear mixed effect modeling tool nlmixr. *CPT Pharmacometrics Syst. Pharmacol.* 8, 923–930 '
                      '(2019).',
    'Dykstra2015': 'Dykstra, K. et al. Reporting guidelines for population pharmacokinetic analyses. *J. '
                   'Pharmacokinet. Pharmacodyn.* 42, 301–314 (2015).',
    'Fidler2019': 'Fidler, M. et al. Nonlinear mixed-effects model development and simulation using nlmixr and '
                  'related R open-source packages. *CPT Pharmacometrics Syst. Pharmacol.* 8, 621–633 (2019).',
    'Eleveld2026': 'Eleveld, D.J., Koomen, J.V., Stevens, J., Colin, P.J. & Struys, M.M.R.F. OpenPMX software for '
                   'nonlinear mixed-effect models in pharmacometrics: precision compared with NONMEM first-order '
                   'conditional estimation. *CPT Pharmacometrics Syst. Pharmacol.* 15, e70250 (2026).',
    'Boeckmann1994': 'Boeckmann, A.J., Sheiner, L.B. & Beal, S.L. *NONMEM Users Guide Part V: Introductory Guide* '
                     '(NONMEM Project Group, University of California, San Francisco, 1994).',
    'nlme': 'Pinheiro, J., Bates, D. & R Core Team. nlme: linear and nonlinear mixed effects models. R package version '
            '3.1-168 <https://CRAN.R-project.org/package=nlme> (2025). Accessed 1 October 2026.',
}
