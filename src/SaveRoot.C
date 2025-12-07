/*----------------------------------------------------------------+
| Macro to Format G4outfiles and create MLinput rootfiles         |
| ML input: (E, px, py, pz, dE/dx) for scat pi, decay p, decay pi |
| truncated dE/dx will be calculated based on parameters below    |
| user parameters: TRUNCATE_RATE (0~1) / MIN_HITS                 |
|                                                                 |
| created 2025-12-06                                              |
+----------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TParticle.h"
#include "TVector3.h"
#include "TDatabasePDG.h"
#include "../S2SG4/include/DetectorID.hh"

const double TRUNCATE_RATE = 0.3;
const int MIN_HITS = 4;

struct HitInfo {
    int trackID;
    int pdg;
    double time;
    double edep;
    TVector3 pos;
    TVector3 mom;
};

struct TrackFeature {
    double E;
    double px;
    double py;
    double pz;
    double dedx;
    double p_mag;
    // sort in p size
    bool operator>(const TrackFeature& other) const { return p_mag > other.p_mag; }
};

// Calculate features
TrackFeature CalculateFeatures(std::vector<HitInfo>& hits, TDatabasePDG* pdgDB) {
    // sort hit time
    std::sort(hits.begin(), hits.end(), [](const HitInfo& a, const HitInfo& b) {
        return a.time < b.time;
    });

    double mass = 9999.9;
    if (pdgDB->GetParticle(hits[0].pdg)) {
        mass = pdgDB->GetParticle(hits[0].pdg)->Mass();
    }

    TVector3 p_vec = hits[0].mom;
    double E = std::sqrt(p_vec.Mag2() + mass * mass);

    // dE/dx (Truncated Mean)
    std::vector<double> dedx_samples;
    for (size_t i = 0; i < hits.size() - 1; ++i) {
        double dx = (hits[i+1].pos - hits[i].pos).Mag();
        if (dx > 0.1) {
            dedx_samples.push_back(hits[i].edep / dx);
        }
    }
    
    double dedx_val = 0.0;
    if (!dedx_samples.empty()) {
        std::sort(dedx_samples.begin(), dedx_samples.end());
        int n_use = (int)(dedx_samples.size() * (1.0 - TRUNCATE_RATE));
        if (n_use < 1) n_use = 1;
        double sum = 0.0;
        for (int i = 0; i < n_use; ++i) sum += dedx_samples[i];
        dedx_val = sum / n_use;
    }

    TrackFeature tf;
    tf.E = E; tf.px = p_vec.Px(); tf.py = p_vec.Py(); tf.pz = p_vec.Pz();
    tf.dedx = dedx_val; tf.p_mag = p_vec.Mag();
    return tf;
}

void SaveRoot(TString inputFile, TString outputFile, int label_val) {
    TFile* f = TFile::Open(inputFile);

    TTreeReader reader("g4s2s", f);
    TTreeReaderValue<std::vector<TParticle>> tpcHits(reader, "TPC");
    TTreeReaderValue<Int_t> mt(reader, "Mt");
    TTreeReaderValue<std::vector<bool>> trig(reader, "trig");

    TFile* fout = new TFile(outputFile, "RECREATE");
    TTree* tree = new TTree("tree", "Training Data");

    Int_t b_label;
    Float_t b_t0_E, b_t0_px, b_t0_py, b_t0_pz, b_t0_dedx;
    Float_t b_t1_E, b_t1_px, b_t1_py, b_t1_pz, b_t1_dedx;
    Float_t b_t2_E, b_t2_px, b_t2_py, b_t2_pz, b_t2_dedx;

    tree->Branch("label", &b_label, "label/I");
    
    // t0 (Fastest)
    tree->Branch("t0_E",   &b_t0_E,   "t0_E/F");
    tree->Branch("t0_px",  &b_t0_px,  "t0_px/F");
    tree->Branch("t0_py",  &b_t0_py,  "t0_py/F");
    tree->Branch("t0_pz",  &b_t0_pz,  "t0_pz/F");
    tree->Branch("t0_dedx",&b_t0_dedx,"t0_dedx/F");

    // t1 (Middle)
    tree->Branch("t1_E",  &b_t1_E,  "t1_E/F");
    tree->Branch("t1_px", &b_t1_px, "t1_px/F");
    tree->Branch("t1_py", &b_t1_py, "t1_py/F");
    tree->Branch("t1_pz", &b_t1_pz, "t1_pz/F");
    tree->Branch("t1_dedx",&b_t1_dedx,"t1_dedx/F");
    
    // t2 (Slowest)
    tree->Branch("t2_E",  &b_t2_E,  "t2_E/F");
    tree->Branch("t2_px", &b_t2_px, "t2_px/F");
    tree->Branch("t2_py", &b_t2_py, "t2_py/F");
    tree->Branch("t2_pz", &b_t2_pz, "t2_pz/F");
    tree->Branch("t2_dedx",&b_t2_dedx,"t2_dedx/F");

    TDatabasePDG* pdgDB = TDatabasePDG::Instance();
    int savedEvents = 0;
    b_label = label_val;

    while (reader.Next()) {
        // Trigger Check
        bool isPiTrigger = (*trig)[kE90TOF] && (*trig)[kE90SAC] && (*trig)[kE90AC1];
        if (!isPiTrigger) continue;

        // Mt Check
        if (*mt != 2) continue;

        std::map<int, std::vector<HitInfo>> trackMap;

        // Hit Collection
        for (const auto& hit : *tpcHits) {
            int tid = hit.GetMother(1); 
            int pdg = hit.GetPdgCode();

            HitInfo h;
            h.trackID = tid;
            h.pdg = pdg;
            h.time = hit.T();
            h.edep = hit.Energy();
            h.pos.SetXYZ(hit.Vx(), hit.Vy(), hit.Vz());
            h.mom.SetXYZ(hit.Px(), hit.Py(), hit.Pz());
            trackMap[tid].push_back(h);
        }

        std::vector<TrackFeature> allTracks;

        for (auto& pair : trackMap) {
            std::vector<HitInfo>& hits = pair.second;
            // nhit cut
            if (hits.size() < MIN_HITS) continue;

            allTracks.push_back(CalculateFeatures(hits, pdgDB));
        }

        // Save ntracks with Mt=2 + PrimPi = 3
        if (allTracks.size() == 3) {
            // sort with momentum value (t0:Fast -> t2:Slow)
            std::sort(allTracks.begin(), allTracks.end(), std::greater<TrackFeature>());

            // t0
            b_t0_E    = allTracks[0].E;
            b_t0_px   = allTracks[0].px;
            b_t0_py   = allTracks[0].py;
            b_t0_pz   = allTracks[0].pz;
            b_t0_dedx = allTracks[0].dedx;

            // t1
            b_t1_E    = allTracks[1].E;
            b_t1_px   = allTracks[1].px;
            b_t1_py   = allTracks[1].py;
            b_t1_pz   = allTracks[1].pz;
            b_t1_dedx = allTracks[1].dedx;

            // t2
            b_t2_E    = allTracks[2].E;
            b_t2_px   = allTracks[2].px;
            b_t2_py   = allTracks[2].py;
            b_t2_pz   = allTracks[2].pz;
            b_t2_dedx = allTracks[2].dedx;

            tree->Fill();
            savedEvents++;
        }
    }

    tree->Write();
    fout->Close();
    f->Close();
    std::cout << "Saved " << savedEvents << " events to " << outputFile << std::endl;
}
