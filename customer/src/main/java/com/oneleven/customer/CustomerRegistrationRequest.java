package com.oneleven.customer;

public record CustomerRegistrationRequest(
        String firstName,
        String lastName,
        String email
) { }
