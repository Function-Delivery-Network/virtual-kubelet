FROM gcr.io/distroless/base

ENV APISERVER_CERT_LOCATION /vkubelet-fdn-0-crt.pem
ENV APISERVER_KEY_LOCATION /vkubelet-fdn-0-key.pem
ENV KUBELET_PORT 10250

# Use the pre-built binary in "bin/virtual-kubelet".
COPY bin/e2e/virtual-kubelet /virtual-kubelet
# Copy the configuration file for the fdn provider.
COPY ./vkubelet-fdn-0-cfg.json /vkubelet-fdn-0-cfg.json
# Copy the certificate for the HTTPS server.
COPY ./vkubelet-fdn-0-crt.pem /vkubelet-fdn-0-crt.pem
# Copy the private key for the HTTPS server.
COPY ./vkubelet-fdn-0-key.pem /vkubelet-fdn-0-key.pem

CMD ["/virtual-kubelet"]
