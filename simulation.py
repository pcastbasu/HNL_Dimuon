import siren
from siren.SIREN_Controller import SIREN_Controller
from siren.SIREN_DarkNews import PyDarkNewsInteractionCollection
import numpy as np

#The goal of this script is to iterate over different HNL masses and mixings and run SIREN simulations for each iteration.

masses  = ["0001000","0002000","0003000","0004000","0005000","0006000","0007000","0008000","0009000","0010000","0020000","0030000","0040000","0050000","0060000","0070000","0080000","0090000"] #in GeV
mixings = np.logspace(-1,-6,num=6)

for m4 in masses:
  for umu4 in mixings:
    events_to_inject = int(1e4) #how many events we simulate per iteration
    experiment       = "IceCube" #what detector density we use
    controller       = SIREN_Controller(events_to_inject, experiment)
    model_kwargs     = {
      "m4": float(m4)*1e-3, #set HNL mass
      "mu_tr_mu4": 0, 
      "UD4": 0,
      "Umu4": umu4, #set mixing
      "epsilon": 0.0,
      "gD": 0.0,
      "decay_product": "mu+mu-",
      "noHC": True,
      "HNLtype": "dirac",
      "nu_flavors":["nu_mu"] #specify neutrino flavor we inject for DarkNews
    }
    primary_type = siren.dataclasses.Particle.ParticleType.NuMu #for injecting muon neutrinos only
    xs_path      = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
    table_dir    = os.path.join(xs_path,"Minimal_M%5.5e_mu%5.5e" % (model_kwargs["m4"], model_kwargs["Umu4"]),) #mmight need to change path if tables already filled
    
    controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs, upscattering=False,fid_vol_secondary=False)
    
    cross_section_model = "HNLDISSplines"
    xsfiledir           = siren.utilities.get_cross_section_model_path(cross_section_model)
    target_type         = siren.dataclasses.Particle.ParticleType.Nucleon
    DIS_xs              = siren.interactions.HNLDISFromSpline(
                                  os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-nu-N-nc-GRV98lo_patched_central.fits"),
                                  os.path.join(xsfiledir, "M_%sMeV/sigma-nu-N-nc-GRV98lo_patched_central.fits"%m4),
                                  model_kwargs["m4"],[0,model_kwargs["Umu4"],0],[primary_type],[target_type],)
    emin                       = max(1e1,1.1*DIS_xs.InteractionThreshold(siren.dataclasses.InteractionRecord()))
    DIS_interaction_collection = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
    
    controller.SetInteractions(DIS_interaction_collection)

    primary_injection_distributions = {}
    primary_physical_distributions  = {}

    # energy distribution
    edist = siren.distributions.PowerLaw(2,emin,1.7e4) #change to PowerLaw(2.58, emin, 1.7e4) for astrophysical neutrinos
    primary_injection_distributions["energy"] = edist  #for astro neutrinos, add normalization to Power Law: norm = 1e-18 * 1.68 *1e4 * 4 * np.pi # GeV^-1 m^-2 s^-1,edist.SetNormalizationAtEnergy(norm,1e5), primary_physical_distributions["energy"] = edist

    # direction distribution
    direction_distribution = siren.distributions.IsotropicDirection()
    primary_injection_distributions["direction"] = direction_distribution #for astro, add primary_physical_distributions["direction"] = direction_distribution
    if primary_type==siren.dataclasses.Particle.ParticleType.NuMu:
      primary_physical_distributions["energy_direction"] = siren.distributions.Tabulated2DFluxDistribution(1e1,1e5,"daemonflux_numu.txt",True) #because this case is handling atmos flux
    elif primary_type==siren.dataclasses.Particle.ParticleType.NuMuBar:
      primary_physical_distributions["energy_direction"] = siren.distributions.Tabulated2DFluxDistribution(1e1,1e5,"daemonflux_numubar.txt",True)

    #position distribution
    muon_range_func       = siren.distributions.LeptonDepthFunction()
    position_distribution = siren.distributions.ColumnDepthPositionDistribution(250.0, 400.0, muon_range_func, set(controller.GetDetectorModelTargets()[0])) #change to 600,600 for full IceCube detector; 250, 400 are for DeepCore. Corresponding changes in the IceCube-v1.dat file in SIREN are also needed.
    primary_injection_distributions["position"]  = position_distribution

    controller.SetProcesses(primary_type, primary_injection_distributions, primary_physical_distributions)
    controller.Initialize()

    def stop(datum, i): #Process continued by siren unless the secondary particle is an HNL
            secondary_type = datum.record.signature.secondary_types[i]
            return secondary_type != siren.dataclasses.Particle.ParticleType.N4

    controller.SetInjectorStoppingCondition(stop)
    events = controller.GenerateEvents(fill_tables_at_exit=False)
    os.makedirs("simulations/dc_atmosnumu/output%5.5e"%model_kwargs["m4"], exist_ok=True) #because this case is handling atmos flux of NuMus in DeepCore
    controller.SaveEvents("simulations/dc_atmosnumu/output%5.5e/test%5.5e" % (model_kwargs["m4"], umu4),fill_tables_at_exit=False)
