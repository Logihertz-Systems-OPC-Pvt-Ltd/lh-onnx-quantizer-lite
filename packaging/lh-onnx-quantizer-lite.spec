Name:           lh-onnx-quantizer-lite
Version:        0.1.0
Release:        1%{?dist}
Summary:        lh-onnx-quant - Kullback-Leibler Divergence Entropy Minimization Calibration
License:        Apache-2.0
URL:            https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-onnx-quantizer-lite
Source0:        %{name}-%{version}-src.tar.gz
Packager:       Sameer Tawade <ops@logihertz.com>
BuildArch:      noarch

%description
Zero-dependency Edge Vision INT4/INT8 Post-Training Quantization (PTQ) engine with Kullback-Leibler (KL) divergence entropy calibration, analytical Conv2D+BatchNorm folding, QDQ synthesis, and SQNR posture auditor.

%prep
%autosetup -n %{name}-0.1.0

%build
# Industrial clean-room build with zero external runtime dependencies

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_docdir}/%{name}
mkdir -p %{buildroot}%{_mandir}/man1

if [ -f bin/lh-onnx-quant ]; then
    install -m 0755 bin/lh-onnx-quant %{buildroot}%{_bindir}/lh-onnx-quant
elif [ -f lh-onnx-quant ]; then
    install -m 0755 lh-onnx-quant %{buildroot}%{_bindir}/lh-onnx-quant
fi

install -m 0644 README.md %{buildroot}%{_docdir}/%{name}/README.md
install -m 0644 SPEC.md %{buildroot}%{_docdir}/%{name}/SPEC.md
install -m 0644 LICENSE %{buildroot}%{_docdir}/%{name}/LICENSE
install -m 0644 NOTICE %{buildroot}%{_docdir}/%{name}/NOTICE

%files
%doc %{_docdir}/%{name}/README.md
%doc %{_docdir}/%{name}/SPEC.md
%doc %{_docdir}/%{name}/NOTICE
%license %{_docdir}/%{name}/LICENSE

%changelog
* Mon Nov 16 2026 Sameer Tawade <ops@logihertz.com> - 0.1.0-1
- Initial production release for lh-onnx-quantizer-lite.
