#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Number of events to read from each input sample.
const int target_sigmaNCusp = 76000;
const int target_qfLambda = 155591;
const int target_qfSigmaZ = 43685;

struct EventVars {
  Int_t label = 0;
  Float_t mm = 0;
  Float_t t0_ux = 0;
  Float_t t0_uy = 0;
  Float_t t0_uz = 0;
  Float_t t0_dedx = 0;
  Float_t t1_ux = 0;
  Float_t t1_uy = 0;
  Float_t t1_uz = 0;
  Float_t t1_dedx = 0;
  Float_t t2_ux = 0;
  Float_t t2_uy = 0;
  Float_t t2_uz = 0;
  Float_t t2_dedx = 0;
};

void SetBranchAddresses(TTree* tree, EventVars& vars) {
  tree->SetBranchAddress("label", &vars.label);
  tree->SetBranchAddress("mm", &vars.mm);
  tree->SetBranchAddress("t0_ux", &vars.t0_ux);
  tree->SetBranchAddress("t0_uy", &vars.t0_uy);
  tree->SetBranchAddress("t0_uz", &vars.t0_uz);
  tree->SetBranchAddress("t0_dedx", &vars.t0_dedx);
  tree->SetBranchAddress("t1_ux", &vars.t1_ux);
  tree->SetBranchAddress("t1_uy", &vars.t1_uy);
  tree->SetBranchAddress("t1_uz", &vars.t1_uz);
  tree->SetBranchAddress("t1_dedx", &vars.t1_dedx);
  tree->SetBranchAddress("t2_ux", &vars.t2_ux);
  tree->SetBranchAddress("t2_uy", &vars.t2_uy);
  tree->SetBranchAddress("t2_uz", &vars.t2_uz);
  tree->SetBranchAddress("t2_dedx", &vars.t2_dedx);
}

void BookOutputBranches(TTree& tree, EventVars& vars) {
  tree.Branch("label", &vars.label, "label/I");
  tree.Branch("mm", &vars.mm, "mm/F");
  tree.Branch("t0_ux", &vars.t0_ux, "t0_ux/F");
  tree.Branch("t0_uy", &vars.t0_uy, "t0_uy/F");
  tree.Branch("t0_uz", &vars.t0_uz, "t0_uz/F");
  tree.Branch("t0_dedx", &vars.t0_dedx, "t0_dedx/F");
  tree.Branch("t1_ux", &vars.t1_ux, "t1_ux/F");
  tree.Branch("t1_uy", &vars.t1_uy, "t1_uy/F");
  tree.Branch("t1_uz", &vars.t1_uz, "t1_uz/F");
  tree.Branch("t1_dedx", &vars.t1_dedx, "t1_dedx/F");
  tree.Branch("t2_ux", &vars.t2_ux, "t2_ux/F");
  tree.Branch("t2_uy", &vars.t2_uy, "t2_uy/F");
  tree.Branch("t2_uz", &vars.t2_uz, "t2_uz/F");
  tree.Branch("t2_dedx", &vars.t2_dedx, "t2_dedx/F");
}

void mktestdata() {
  const std::string tree_name = "g4s2s";
  const std::string output_path = "../data/input/test.root";

  struct Sample {
    std::string name;
    std::string path;
    Long64_t target = 0;
    std::unique_ptr<TFile> file;
    TTree* tree = nullptr;
    Long64_t entries = 0;
    std::vector<Long64_t> picked;
  };

  std::vector<Sample> samples;
  samples.reserve(3);
  samples.push_back({"SigmaNCusp", "../data/input/SigmaNCusp_test.root", target_sigmaNCusp});
  samples.push_back({"QFLambda", "../data/input/QFLambda_test.root", target_qfLambda});
  samples.push_back({"QFSigmaZ", "../data/input/QFSigmaZ_test.root", target_qfSigmaZ});

  std::mt19937 rng(static_cast<unsigned int>(
      std::chrono::steady_clock::now().time_since_epoch().count()));

  EventVars vars;

  for (auto& s : samples) {
    s.file.reset(TFile::Open(s.path.c_str(), "READ"));
    s.tree = dynamic_cast<TTree*>(s.file->Get(tree_name.c_str()));
    SetBranchAddresses(s.tree, vars);
    s.entries = s.tree->GetEntries();

    const Long64_t desired = s.target;

    std::vector<Long64_t> indices(s.entries);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    if (desired <= s.entries) {
      indices.resize(static_cast<size_t>(desired));
      s.picked = std::move(indices);
    } else {
      s.picked = std::move(indices);
      std::uniform_int_distribution<Long64_t> dist(0, s.entries - 1);
      while (s.picked.size() < static_cast<size_t>(desired)) {
        s.picked.push_back(dist(rng));
      }
      std::shuffle(s.picked.begin(), s.picked.end(), rng);
    }

    std::cout << s.name << ": target=" << desired << " available=" << s.entries
              << " selected=" << s.picked.size();
    if (desired > s.entries) {
      std::cout << " (oversampled to reach target)";
    }
    std::cout << std::endl;
  }

  struct EntryRef {
    TTree* tree = nullptr;
    Long64_t entry = 0;
  };

  std::vector<EntryRef> combined;
  size_t total_entries = 0;
  for (const auto& s : samples) {
    total_entries += s.picked.size();
  }
  combined.reserve(total_entries);

  for (const auto& s : samples) {
    for (auto idx : s.picked) {
      combined.push_back({s.tree, idx});
    }
  }

  std::shuffle(combined.begin(), combined.end(), rng);

  TFile output(output_path.c_str(), "RECREATE");
  TTree out_tree(tree_name.c_str(), "Shuffled test sample (event-count based)");
  BookOutputBranches(out_tree, vars);

  for (const auto& ref : combined) {
    ref.tree->GetEntry(ref.entry);
    out_tree.Fill();
  }

  output.cd();
  out_tree.Write();
  output.Close();

  std::cout << "Wrote " << out_tree.GetEntries() << " entries to " << output_path << std::endl;
}
