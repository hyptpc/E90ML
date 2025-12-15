#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct ReactionSpec {
  int label;
  std::string name;
  Color_t color;
};

void plotMM(const char* inputPath = "../data/output/test.root", const char* treeName = "g4s2s",
            const char* outputPath = "") {
  gStyle->SetOptStat(0);
  const int font = 132;  // Times New Roman
  gStyle->SetTextFont(font);
  gStyle->SetLabelFont(font, "XYZ");
  gStyle->SetTitleFont(font, "XYZ");
  gStyle->SetTitleFont(font, "");
  gStyle->SetLabelSize(0.04, "XYZ");
  gStyle->SetTitleSize(0.04, "XYZ");
  gStyle->SetTitleSize(0.06, "t");

  std::unique_ptr<TFile> file(TFile::Open(inputPath, "READ"));
  TTree* tree = dynamic_cast<TTree*>(file->Get(treeName));

  const double mmMin = 2050.0;
  const double mmMax = 2250.0;
  const double binWidth = 0.4;
  const int nBins = static_cast<int>((mmMax - mmMin) / binWidth);

  auto makeHist = [&](const std::string& name, Color_t color, int width = 2) {
    TH1D* h = new TH1D(name.c_str(), "", nBins, mmMin, mmMax);
    h->Sumw2();
    h->SetLineColor(color);
    h->SetLineWidth(width);
    h->SetFillStyle(0);
    h->GetXaxis()->CenterTitle(true);
    h->GetYaxis()->CenterTitle(true);
    h->GetXaxis()->SetTitle("Mass [MeV]");
    h->GetYaxis()->SetTitle("Counts / 0.4 MeV");
    return h;
  };

  std::vector<ReactionSpec> reactions = {
      {1, "#SigmaN Cusp", kMagenta},
      {2, "QF#Lambda", kRed},
      {3, "QF#Sigma^{0}", kGreen},
  };

  TH1D* hRawTotal = makeHist("h_raw_total", kBlack, 3);
  hRawTotal->SetTitle("Missing Mass (raw);Mass [MeV];Counts / 0.4 MeV");
  tree->Draw(Form("mm>>%s", hRawTotal->GetName()), "", "goff");

  TH1D* hSelTotal = makeHist("h_sel_total", kBlack, 3);
  hSelTotal->SetTitle("Missing Mass (signal-selected);Mass [MeV];Counts / 0.4 MeV");
  tree->Draw(Form("mm>>%s", hSelTotal->GetName()), "out==1", "goff");

  std::vector<TH1D*> rawHists;
  std::vector<TH1D*> selHists;
  rawHists.reserve(reactions.size());
  selHists.reserve(reactions.size());

  for (const auto& r : reactions) {
    TH1D* hRaw = makeHist(Form("h_raw_%s", r.name.c_str()), r.color);
    tree->Draw(Form("mm>>%s", hRaw->GetName()), Form("label==%d", r.label), "goff");
    rawHists.push_back(hRaw);

    TH1D* hSel = makeHist(Form("h_sel_%s", r.name.c_str()), r.color);
    tree->Draw(Form("mm>>%s", hSel->GetName()), Form("out==1 && label==%d", r.label), "goff");
    selHists.push_back(hSel);
  }

  auto setMaxFrom = [](TH1D* total, const std::vector<TH1D*>& parts) {
    double maxVal = total->GetMaximum();
    for (auto* h : parts) {
      maxVal = std::max(maxVal, h->GetMaximum());
    }
    total->SetMaximum(maxVal * 1.25);
  };
  setMaxFrom(hRawTotal, rawHists);
  setMaxFrom(hSelTotal, selHists);

  TCanvas* c = new TCanvas("c_mm", "mm comparison", 1200, 600);
  c->Divide(2, 1);

  c->cd(1);
  gPad->SetTicks(1, 1);
  gPad->SetLeftMargin(0.15);
  hRawTotal->Draw("hist");
  for (auto* h : rawHists) {
    h->Draw("hist same");
  }
  hRawTotal->Draw("hist same");
  TLegend* legRaw = new TLegend(0.65, 0.60, 0.88, 0.88);
  legRaw->SetBorderSize(0);
  legRaw->AddEntry(hRawTotal, "Total", "l");
  for (size_t i = 0; i < reactions.size(); ++i) {
    legRaw->AddEntry(rawHists[i], reactions[i].name.c_str(), "l");
  }
  legRaw->Draw();

  c->cd(2);
  gPad->SetTicks(1, 1);
  gPad->SetLeftMargin(0.15);
  hSelTotal->Draw("hist");
  for (auto* h : selHists) {
    h->Draw("hist same");
  }
  hSelTotal->Draw("hist same");
  TLegend* legSel = new TLegend(0.65, 0.60, 0.88, 0.88);
  legSel->SetBorderSize(0);
  legSel->AddEntry(hSelTotal, "Total (out==1)", "l");
  for (size_t i = 0; i < reactions.size(); ++i) {
    legSel->AddEntry(selHists[i], reactions[i].name.c_str(), "l");
  }
  legSel->Draw();

  if (std::string(outputPath).size() > 0) {
    c->SaveAs(outputPath);
    std::cout << "Saved canvas to " << outputPath << std::endl;
  }
}
